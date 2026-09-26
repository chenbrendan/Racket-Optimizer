import unittest
import numpy as np
import racket_fea as rf

class RacketFEATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame=rf.RacketFrameFEA(n_nodes=80)
        cls.strings=rf.build_string_grommets(cls.frame,16,18,tension=100.)
        cls.u,cls.m=rf.build_influence_matrices(cls.frame,cls.strings)
        loads=np.array([s.target_tension for s in cls.strings])
        cls.u*=loads[:,None]; cls.m*=loads[:,None,None]

    def test_unique_grommets(self):
        self.assertEqual(len({(s.node_a,s.node_b) for s in self.strings}),len(self.strings))

    def test_baseline_complete_and_feasible(self):
        seq=rf.generate_standard_baseline(self.strings)
        self.assertEqual(sorted(seq),list(range(len(self.strings))))
        self.assertTrue(rf.sequence_is_feasible(seq,self.strings))

    def test_final_state_order_independent(self):
        np.testing.assert_allclose(np.sum(self.u,axis=0),np.sum(self.u[::-1],axis=0),rtol=1e-12,atol=1e-12)

    def test_invalid_sequence_rejected(self):
        with self.assertRaises(ValueError): rf.evaluate_sequence(self.frame,self.u,self.m,list(range(10)))

    def test_optimizer_is_feasible_and_non_worse(self):
        baseline=rf.generate_standard_baseline(self.strings)
        base=rf.evaluate_sequence(self.frame,self.u,self.m,baseline)
        seq,result,_=rf.optimize_sequence(self.frame,self.strings,self.u,self.m,iterations=100,restarts=1,seed=2)
        self.assertTrue(rf.sequence_is_feasible(seq,self.strings))
        self.assertLessEqual(result.cost,base.cost+1e-12)

if __name__ == "__main__": unittest.main()
