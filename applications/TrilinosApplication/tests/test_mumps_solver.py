import pathlib

import KratosMultiphysics
import KratosMultiphysics.TrilinosApplication as KratosTrilinos
import KratosMultiphysics.KratosUnittest as KratosUnittest
from KratosMultiphysics.TrilinosApplication import trilinos_linear_solver_factory


def GetFilePath(fileName):
    return str(pathlib.Path(__file__).absolute().parent / fileName)


@KratosUnittest.skipUnless(
    hasattr(KratosTrilinos, 'TrilinosMumpsSolver'),
    "TrilinosMumpsSolver not compiled (requires TRILINOS_APPLICATION_USE_MUMPS_DIRECTLY=ON).")
class TestMumpsDirectLinearSolvers(KratosUnittest.TestCase):
    """Tests for TrilinosMumpsSolver — the direct MUMPS wrapper that bypasses Amesos/Amesos2."""

    def _GetDataComm(self):
        return KratosMultiphysics.Testing.GetDefaultDataCommunicator()

    def _BuildSystem(self, matrix_name="A.mm"):
        """Return (space, comm, pA, pAcopy, pb, px, pbcopy) ready for a solve."""
        data_comm = self._GetDataComm()
        comm = KratosTrilinos.CreateEpetraCommunicator(data_comm)
        space = KratosTrilinos.TrilinosSparseSpace()

        path = GetFilePath("auxiliary_files/matrix_market_files/" + matrix_name)
        pA = space.ReadMatrixMarketMatrix(path, comm)
        pAcopy = space.ReadMatrixMarketMatrix(path, comm)
        n = space.Size1(pA.GetReference())

        pb = space.CreateEmptyVectorPointer(comm)
        space.ResizeVector(pb, n)
        space.SetToZeroVector(pb.GetReference())
        space.SetValue(pb.GetReference(), 0, 1.0)

        pbcopy = space.CreateEmptyVectorPointer(comm)
        space.ResizeVector(pbcopy, n)
        space.SetToZeroVector(pbcopy.GetReference())
        space.UnaliasedAdd(pbcopy.GetReference(), 1.0, pb.GetReference())

        px = space.CreateEmptyVectorPointer(comm)
        space.ResizeVector(px, n)
        space.SetToZeroVector(px.GetReference())

        return space, comm, data_comm, pA, pAcopy, pb, pbcopy, px

    def _CheckResidual(self, space, pAcopy, pbcopy, px, data_comm, tolerance=1e-9):
        n = space.Size1(pAcopy.GetReference())
        ptmp = space.CreateEmptyVectorPointer(
            KratosTrilinos.CreateEpetraCommunicator(data_comm))
        space.ResizeVector(ptmp, n)
        space.SetToZeroVector(ptmp.GetReference())
        space.Mult(pAcopy.GetReference(), px.GetReference(), ptmp.GetReference())

        pcheck = space.CreateEmptyVectorPointer(
            KratosTrilinos.CreateEpetraCommunicator(data_comm))
        space.ResizeVector(pcheck, n)
        space.SetToZeroVector(pcheck.GetReference())
        space.UnaliasedAdd(pcheck.GetReference(), 1.0, pbcopy.GetReference())
        space.UnaliasedAdd(pcheck.GetReference(), -1.0, ptmp.GetReference())

        achieved_norm = space.TwoNorm(pcheck.GetReference())
        nproc = data_comm.Size()
        target_norm = tolerance * space.TwoNorm(pbcopy.GetReference()) * nproc
        return achieved_norm, target_norm

    # ------------------------------------------------------------------
    # Basic correctness test (unsymmetric, sym=0)
    # ------------------------------------------------------------------
    def test_mumps_direct_unsymmetric(self):
        """Solve using the default (unsymmetric) setting and verify the residual."""
        settings = KratosMultiphysics.Parameters('{"solver_type": "mumps_direct", "sym": 0}')
        space, comm, data_comm, pA, pAcopy, pb, pbcopy, px = self._BuildSystem()

        solver = trilinos_linear_solver_factory.ConstructSolver(settings)
        solver.Solve(pA.GetReference(), px.GetReference(), pb.GetReference())

        achieved_norm, target_norm = self._CheckResidual(space, pAcopy, pbcopy, px, data_comm)
        self.assertLessEqual(achieved_norm, target_norm)

    # ------------------------------------------------------------------
    # Symmetric solve (sym=1)
    # ------------------------------------------------------------------
    def test_mumps_direct_symmetric(self):
        """Solve with sym=1 (symmetric but not necessarily SPD)."""
        settings = KratosMultiphysics.Parameters('{"solver_type": "mumps_direct", "sym": 1}')
        space, comm, data_comm, pA, pAcopy, pb, pbcopy, px = self._BuildSystem()

        solver = trilinos_linear_solver_factory.ConstructSolver(settings)
        solver.Solve(pA.GetReference(), px.GetReference(), pb.GetReference())

        achieved_norm, target_norm = self._CheckResidual(space, pAcopy, pbcopy, px, data_comm)
        self.assertLessEqual(achieved_norm, target_norm)

    # ------------------------------------------------------------------
    # PrintInfo must not raise
    # ------------------------------------------------------------------
    def test_mumps_direct_print_info(self):
        """TrilinosMumpsSolver.PrintInfo must not raise an exception."""
        settings = KratosMultiphysics.Parameters('{"solver_type": "mumps_direct"}')
        solver = trilinos_linear_solver_factory.ConstructSolver(settings)
        try:
            solver.PrintInfo()
        except Exception as exc:
            self.fail(f"PrintInfo raised an unexpected exception: {exc}")

    # ------------------------------------------------------------------
    # Second solve reuses symbolic analysis (mReanalyze = false path)
    # ------------------------------------------------------------------
    def test_mumps_direct_repeat_solve_same_result(self):
        """Two consecutive Solve calls on the same matrix must give identical solutions."""
        settings = KratosMultiphysics.Parameters('{"solver_type": "mumps_direct", "sym": 0}')
        space, comm, data_comm, pA, pAcopy, pb, pbcopy, px = self._BuildSystem()

        solver = trilinos_linear_solver_factory.ConstructSolver(settings)

        # First solve
        solver.Solve(pA.GetReference(), px.GetReference(), pb.GetReference())
        x1_norm = space.TwoNorm(px.GetReference())

        # Restore rhs and solution vector for second solve
        n = space.Size1(pA.GetReference())
        pb2 = space.CreateEmptyVectorPointer(comm)
        space.ResizeVector(pb2, n)
        space.SetToZeroVector(pb2.GetReference())
        space.SetValue(pb2.GetReference(), 0, 1.0)

        px2 = space.CreateEmptyVectorPointer(comm)
        space.ResizeVector(px2, n)
        space.SetToZeroVector(px2.GetReference())

        # Second solve — symbolic analysis is skipped (mReanalyze = false internally)
        solver.Solve(pA.GetReference(), px2.GetReference(), pb2.GetReference())
        x2_norm = space.TwoNorm(px2.GetReference())

        # Both solutions should have the same norm (identical system)
        self.assertAlmostEqual(x1_norm, x2_norm, places=10)

        # And the second solution must also satisfy the original system
        achieved_norm, target_norm = self._CheckResidual(space, pAcopy, pbcopy, px2, data_comm)
        self.assertLessEqual(achieved_norm, target_norm)


if __name__ == '__main__':
    KratosMultiphysics.Logger.GetDefaultOutput().SetSeverity(KratosMultiphysics.Logger.Severity.WARNING)
    KratosUnittest.main()
