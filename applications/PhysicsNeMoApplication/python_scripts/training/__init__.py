"""Training loops, datasets and training-time schemes.

======================== ======================================================
Module                   Provides
======================== ======================================================
``training_utils``       ``TrainModel``/``SaveTrainedModel`` - the
                         ``Parameters``-driven loop, callbacks, warm restarts,
                         OOD calibration and FSDP2-safe checkpoint writing
``torch_dataset``        dataset and datapipe factories over exported files
                         (``.npz``, ``.pmsh``, DoMINO/Transolver, augmentation,
                         mixing)
``streaming_dataset``    ``LiveSampleQueue`` and ``StreamingDataset`` - training
                         straight out of a running solve, no file round trip
``temporal_training``    window datasets and ``TrainAutoregressive`` (BPTT
                         through a self-fed rollout)
``diffusion_utils``      ``TrainDiffusionModel``/``TrainCorrDiffPair`` and the
                         EDM sampler wrappers
``domino_finetune``      predictor-corrector and LoRA adaptation of a pretrained
                         DoMINO checkpoint
``rom_temporal``         reduced-trajectory datasets and temporal-attention
                         training in ROM space
``rollout_utils``        ``EvaluateRollout`` - multi-step error growth of a
                         trained time-series surrogate
======================== ======================================================

Nothing here defines a Kratos ``Process``; the process that fills the streaming
queue is ``processes.export.streaming_dataset_export_process``.
"""
