"""Training samples streamed out of a RUNNING Kratos solve.

Today training data takes a detour through disk: DatasetExportProcess writes
one .npz per step and CreateNpzDataset reads them back. That is the right
default (samples are reusable, shuffleable, inspectable), but it forces the
solve to finish before training starts and it writes files nobody keeps.

This module removes the detour. A process pushes each step's gathered fields
into a queue; an iterable dataset drains it and yields exactly the same
(inputs, targets) items CreateNpzDataset produces, so downstream consumers
cannot tell the two apart - which the tests assert directly by running the
same case both ways.

    solve ──StreamingDatasetExportProcess──▶ LiveSampleQueue
                                                  │
                              KratosStreamingDataset ──▶ TrainModel

Four upstream contracts are encoded here, each verified rather than assumed:

- physicsnemo's `IterableDatasetBase` is an ABC, **not** a
  torch.utils.data.IterableDataset, so a bare subclass is rejected by torch's
  DataLoader ("has no len()"). The dataset below inherits BOTH.
- physicsnemo's own DataLoader unpacks every yielded item as
  `(data, metadata)`, which would silently swallow the targets of an
  `(inputs, targets)` tuple. Setting `yields_batches = True` bypasses
  collation entirely and passes items through verbatim - correct here anyway,
  since a solver step already produces a whole batch.
- `num_workers > 0` duplicates an iterable stream across workers, so the
  dataset refuses it rather than training on each sample twice.
- Output buffers must not be reused across yields (the loader may still be
  reading the previous one), so every emission allocates fresh tensors.

A stream is single-pass: `TrainModel`'s "streaming" block pins one epoch,
because a second pass over an exhausted queue would otherwise look like a
zero-loss epoch.

torch/physicsnemo are optional runtime dependencies - imported lazily.
"""

import collections
import threading

import numpy

import KratosMultiphysics as Kratos

from KratosMultiphysics.PhysicsNeMoApplication.utilities.tensor_adaptor_dataset_utils import GetTensorAdaptor

_QUEUE_CLOSED = object()


def _TryImportTorch():
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.streaming_dataset requires torch, which could not be "
            "imported. Install it with e.g. 'pip install torch'.") from e


def _TryImportIterableDatasetBase():
    try:
        from physicsnemo.datapipes import IterableDatasetBase
        return IterableDatasetBase
    except ImportError as e:
        raise ImportError(
            "PhysicsNeMoApplication.streaming_dataset requires physicsnemo >= 2.2 "
            "(IterableDatasetBase landed in 2.2), which could not be imported. Install "
            "it with e.g. 'pip install -U nvidia-physicsnemo'.") from e


def GatherSampleArrays(model_part: Kratos.ModelPart, field_specs) -> dict:
    """One step's fields in DatasetExportProcess's on-disk layout.

    Shared by the file exporter and the live stream so the two cannot drift:
    "TIME"/"STEP" as 0-d arrays, then one entry per field keyed
    "<VARIABLE>__<data_location>". Returns None on non-writing ranks of a
    distributed run, mirroring the exporter.
    """
    step = model_part.ProcessInfo[Kratos.STEP]
    arrays = {
        "TIME": numpy.array(model_part.ProcessInfo[Kratos.TIME]),
        "STEP": numpy.array(step),
    }
    if model_part.IsDistributed():
        from KratosMultiphysics.PhysicsNeMoApplication import distributed_utils
        is_writing_rank = model_part.GetCommunicator().GetDataCommunicator().Rank() == 0
        for variable_name, data_location in field_specs:
            _, values = distributed_utils.GatherFieldToRank0(
                model_part, variable_name, data_location)
            if is_writing_rank:
                arrays[f"{variable_name}__{data_location}"] = values
        if not is_writing_rank:
            return None
    else:
        for variable_name, data_location in field_specs:
            variable = Kratos.KratosGlobals.GetVariable(variable_name)
            tensor_adaptor = GetTensorAdaptor(model_part, data_location, variable)
            arrays[f"{variable_name}__{data_location}"] = numpy.array(tensor_adaptor.data)
    return arrays


class LiveSampleQueue:
    """Bounded hand-off between a running solve and a training loop.

    The solve is the producer and (in-process) the consumer runs after it, so
    this is deliberately simple: a bounded deque plus a close flag. Bounding
    matters because a solve that outruns training would otherwise grow the
    queue without limit; `Push` blocks once `max_size` is reached.

    Closing is explicit and one-way: a consumer cannot otherwise distinguish
    "no sample yet" from "the solve has ended", which is what would make a
    streaming epoch hang.
    """

    def __init__(self, max_size: int = 0):
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0 (0 = unbounded), got {max_size}.")
        self._items = collections.deque()
        self._max_size = max_size
        self._closed = False
        self._condition = threading.Condition()

    def Push(self, sample) -> None:
        """Appends a sample, blocking while the queue is full."""
        with self._condition:
            if self._closed:
                raise RuntimeError("Push after Close: the queue no longer accepts samples.")
            while self._max_size and len(self._items) >= self._max_size:
                self._condition.wait()
            self._items.append(sample)
            self._condition.notify_all()

    def Close(self) -> None:
        """Signals that no further samples will arrive."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    def Drain(self):
        """Yields samples until the queue is closed AND empty."""
        while True:
            with self._condition:
                while not self._items and not self._closed:
                    self._condition.wait()
                if not self._items:
                    return
                sample = self._items.popleft()
                self._condition.notify_all()
            yield sample


def _AssembleSample(arrays, keys, torch):
    """The (n_entities, total_width) float32 block CreateNpzDataset builds."""
    blocks = []
    for key in keys:
        if key not in arrays:
            raise KeyError(
                f"Key \"{key}\" is not in the streamed sample; available: "
                f"{sorted(arrays)}.")
        block = numpy.asarray(arrays[key], dtype=numpy.float32)
        blocks.append(block.reshape(block.shape[0], -1))
    # a fresh allocation per emission: the loader may still be reading the
    # previous one, and upstream warns that reusing output buffers races
    return torch.from_numpy(numpy.ascontiguousarray(
        numpy.concatenate(blocks, axis=-1)))


def CreateStreamingDataset(queue: LiveSampleQueue, input_keys, output_keys):
    """An iterable dataset over a LiveSampleQueue.

    Yields `(inputs, targets)` float32 tensors in exactly the layout
    CreateNpzDataset produces - same per-entity flattening, same
    concatenation order - so a streamed run and a dumped run are
    interchangeable.

    Args:
        queue: The LiveSampleQueue a StreamingDatasetExportProcess fills.
        input_keys: Field keys ("<VARIABLE>__<location>") forming the input.
        output_keys: Field keys forming the training target.

    Returns:
        A dataset that is both a physicsnemo IterableDatasetBase and a
        torch.utils.data.IterableDataset.
    """
    torch = _TryImportTorch()
    IterableDatasetBase = _TryImportIterableDatasetBase()

    input_keys = list(input_keys)
    output_keys = list(output_keys)
    if not input_keys or not output_keys:
        raise ValueError("Both input_keys and output_keys must be non-empty.")

    class KratosStreamingDataset(IterableDatasetBase, torch.utils.data.IterableDataset):
        """Live solver output as a single-pass training stream."""

        # physicsnemo's loader would otherwise unpack each item as
        # (data, metadata) and silently discard the targets
        yields_batches = True

        def __init__(self):
            self._queue = queue
            self._epoch = 0
            self._generator = None
            self._emitted = 0

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None and worker_info.num_workers > 1:
                raise RuntimeError(
                    "KratosStreamingDataset does not support num_workers > 1: every "
                    "worker would replay the whole stream, training on each sample "
                    "once per worker. Use num_workers=0.")
            for arrays in self._queue.Drain():
                self._emitted += 1
                yield (_AssembleSample(arrays, input_keys, torch),
                       _AssembleSample(arrays, output_keys, torch))

        def set_epoch(self, epoch: int) -> None:
            self._epoch = int(epoch)
            self._emitted = 0

        def set_generator(self, generator) -> None:
            self._generator = generator

        def close(self) -> None:
            self._queue.Close()

        @property
        def emitted(self) -> int:
            """Samples yielded so far - the streaming counterpart of len()."""
            return self._emitted

    return KratosStreamingDataset()


def Factory(settings: Kratos.Parameters, model: Kratos.Model):
    if not isinstance(settings, Kratos.Parameters):
        raise TypeError("Expected input shall be a Parameters object, encapsulating a json string")
    return StreamingDatasetExportProcess(model, settings["Parameters"])


class StreamingDatasetExportProcess(Kratos.Process):
    """Pushes each due step's fields into a queue instead of onto disk.

    Same settings and same gather as DatasetExportProcess, minus the paths.
    Attach the queue with SetQueue (or read the one it creates) and hand it
    to CreateStreamingDataset.

    Settings:
        {
            "model_part_name" : "",
            "list_of_fields"  : [ { "variable_name" : "...",
                                    "data_location" : "node_historical" } ],
            "output_interval" : 1,
            "max_queue_size"  : 0,     // 0 = unbounded
            "close_on_finalize" : true // ends the stream at ExecuteFinalize
        }
    """

    def __init__(self, model: Kratos.Model, settings: Kratos.Parameters) -> None:
        super().__init__()

        default_settings = Kratos.Parameters("""{
            "model_part_name"   : "PLEASE_SPECIFY_MODEL_PART_NAME",
            "list_of_fields"    : [
                {
                    "variable_name" : "PLEASE_SPECIFY_VARIABLE_NAME",
                    "data_location" : "node_historical"
                }
            ],
            "output_interval"   : 1,
            "max_queue_size"    : 0,
            "close_on_finalize" : true
        }""")
        settings.ValidateAndAssignDefaults(default_settings)
        for i in range(settings["list_of_fields"].size()):
            settings["list_of_fields"][i].ValidateAndAssignDefaults(
                default_settings["list_of_fields"][0])

        self.model_part = model[settings["model_part_name"].GetString()]
        self.field_specs = [
            (settings["list_of_fields"][i]["variable_name"].GetString(),
             settings["list_of_fields"][i]["data_location"].GetString())
            for i in range(settings["list_of_fields"].size())
        ]
        self.output_interval = settings["output_interval"].GetInt()
        if self.output_interval < 1:
            raise ValueError(
                f"\"output_interval\" must be >= 1 [ output_interval = {self.output_interval} ].")
        self.close_on_finalize = settings["close_on_finalize"].GetBool()
        self.queue = LiveSampleQueue(settings["max_queue_size"].GetInt())

    def SetQueue(self, queue: LiveSampleQueue) -> None:
        """Uses a caller-provided queue instead of the one created here."""
        self.queue = queue

    def ExecuteFinalizeSolutionStep(self) -> None:
        step = self.model_part.ProcessInfo[Kratos.STEP]
        if step % self.output_interval != 0:
            return
        arrays = GatherSampleArrays(self.model_part, self.field_specs)
        if arrays is not None:   # None on non-writing ranks
            self.queue.Push(arrays)

    def ExecuteFinalize(self) -> None:
        if self.close_on_finalize:
            self.queue.Close()
