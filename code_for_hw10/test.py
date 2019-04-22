from dist import *
from mdp10 import *
from util import *



def tinyTerminal(s):
    return s==4
def tinyR(s, a):
    if s == 1: return 1
    elif s == 3: return 2
    else: return 0
def tinyTrans(s, a):
    if s == 0:
        if a == 'a':
            return DDist({1 : 0.9, 2 : 0.1})
        else:
            return DDist({1 : 0.1, 2 : 0.9})
    elif s == 1:
        return DDist({1 : 0.1, 0 : 0.9})
    elif s == 2:
        return DDist({2 : 0.1, 3 : 0.9})
    elif s == 3:
        return DDist({3 : 0.1, 0 : 0.5, 4 : 0.4})
    elif s == 4:
        return DDist({4 : 1.0})


def testQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn(tiny, q)
    return list(qf.q.items())

def testBatchQ_1():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn_batch(tiny, q)
    return list(qf.q.items())

def testBatchQ():
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = TabularQ(tiny.states, tiny.actions)
    qf = Q_learn_batch(tiny, q)
    ret = list(qf.q.items())
    expected = [((0, 'a'), 4.7566600197286535), ((0, 'b'), 3.993296047838986), 
                ((1, 'a'), 5.292467934685342), ((1, 'b'), 5.364014782870985), 
                ((2, 'a'), 4.139537149779127), ((2, 'b'), 4.155347555640753), 
                ((3, 'a'), 4.076532544818926), ((3, 'b'), 4.551442974149778), 
                ((4, 'a'), 0.0), ((4, 'b'), 0.0)]

    ok = True
    for (s,a), v in expected:
      qv = qf.get(s,a)
      if abs(qv-v) > 1.0e-5:
        print("Oops!  For (s=%s, a=%s) expected %s, but got %s" % (s, a, v, qv))
        ok = False
    if ok:
      print("Tests passed!")
      
      return list(qf.q.items())

def test_NNQ_1(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    return [q.get(s,a) for s in q.states for a in q.actions]

def test_NNQ(data):
    tiny = MDP([0, 1, 2, 3, 4], ['a', 'b'], tinyTrans, tinyR, 0.9)
    tiny.terminal = tinyTerminal
    q = NNQ(tiny.states, tiny.actions, tiny.state2vec, 2, 10)
    q.update(data, 1)
    ret =  [q.get(s,a) for s in q.states for a in q.actions]
    expect = [np.array([[-0.07211456]]), np.array([[-0.19553234]]), 
              np.array([[-0.21926211]]), np.array([[0.01699455]]), 
              np.array([[-0.26390356]]), np.array([[0.06374809]]), 
              np.array([[0.0340214]]), np.array([[-0.18334733]]), 
              np.array([[-0.438375]]), np.array([[-0.13844737]])]
    cnt = 0
    ok = True
    for s in q.states:
      for a in q.actions:
        if not np.all(np.abs(ret[cnt]-expect[cnt]) < 1.0e0):
          print("Oops, for s=%s, a=%s expected %s but got %s" % (s, a, expect[cnt], ret[cnt]))
          ok = False
        cnt += 1
    if ok:
      print("Output looks generally ok")
    return q
  
test_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)])

random.seed(0)
#print(testQ())
print(testBatchQ())
#print(test_NNQ([(0,'a',0.3),(1,'a',0.1),(0,'a',0.1),(1,'a',0.5)]))
