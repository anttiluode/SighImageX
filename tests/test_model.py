import unittest
import numpy as np
from model import N,K,block,new_model,forward,loss_gradient
from probe import recover


class Tests(unittest.TestCase):
    def test_writer_gradient_through_energy_and_two_blocks(self):
        rng=np.random.default_rng(81)
        w=new_model()+rng.normal(0,.06,(2,11,10))
        x=rng.normal(0,.3,(2,N,N))
        target=.7*x
        _,g=loss_gradient(x,target,w)
        for index in [(0,3,4),(0,10,2),(1,5,8),(1,10,1)]:
            d=np.zeros_like(w); d[index]=1e-5
            scalar=lambda wt:np.mean((forward(x,wt)[0]-target)**2)
            observed=(scalar(w+d)-scalar(w-d))/2e-5
            self.assertAlmostEqual(observed,g[index],places=8)

    def test_boundary_probe_without_writer_weights(self):
        rng=np.random.default_rng(23)
        x=rng.normal(0,.3,(1,N,N));original=x*.73+.015
        expected=rng.uniform(.06,2.8,(1,K))
        y,_=block(original,x,new_model()[0],expected)
        gains,known,residual=recover(original[0],x[0],y[0])
        self.assertTrue(known.all())
        np.testing.assert_allclose(gains,expected[0],atol=1e-10)
        self.assertLess(residual,1e-11)

    def test_dark_bands_and_wrong_family(self):
        w=new_model()[0]
        for level,indices in [(0,[]),(.24,[0])]:
            x=np.full((1,N,N),float(level))
            y,_=block(x,x,w)
            _,known,_=recover(x[0],x[0],y[0])
            self.assertEqual(np.flatnonzero(known).tolist(),indices)
        x=np.random.default_rng(47).normal(size=(1,N,N))
        y,_=block(x,x,w)
        _,_,res=recover(x[0],x[0],np.roll(y[0],1,axis=0))
        self.assertGreater(res,.1)
