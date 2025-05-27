import unittest
from godot.core.tempo import Epoch
from mani import GodotHandler
from StateEvaluator import StateEvaluator, SEEnum

class TestStateMachine(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
    
    def getSEfromGodot(self, t) -> StateEvaluator:
        godotHandler = GodotHandler(t, t, 30.0, './universe.yml')
        results = godotHandler._evaluate_timestamps([t])
        res = godotHandler._move_to_StateEvaluator(results, t)
        return res

    def testLOS(self):
        t = Epoch('2026-04-02T01:00:00 TDB')
        res = self.getSEfromGodot(t)
        self.assertTrue(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertTrue(res.above_elev('CB11',0.0).item())
    
    def testBlockedGS(self):
        t = Epoch('2026-04-02T17:20:00 TDB')
        res = self.getSEfromGodot(t)
        self.assertTrue(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertFalse(res.above_elev('CB11',0.0).item())

    def testBlockedMoon(self):
        t = Epoch('2026-04-02T01:30:00 TDB')
        res = self.getSEfromGodot(t)
        self.assertFalse(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertTrue(res.above_elev('CB11',0.0).item())

    def testLos2(self):
        t = Epoch('2026-09-02T01:14:01 TDB')
        res = self.getSEfromGodot(t)
        self.assertTrue(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertTrue(res.above_elev('CB11',0.0).item())

    def testAngle(self):
        #Above 10
        t = Epoch('2026-09-02T10:30:01 TDB')
        res = self.getSEfromGodot(t)
        self.assertTrue(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertTrue(res.above_elev('CB11',10.0).item())

        #Below 10
        t = Epoch('2026-09-02T10:40:01 TDB')
        res = self.getSEfromGodot(t)
        self.assertTrue(res.has([SEEnum.CLEAR_MOON_CB]).item())
        self.assertFalse(res.above_elev('CB11',10.0).item())

    def testMultipleStaions(self):
        t = Epoch('2026-04-02T17:20:00 TDB')
        res = self.getSEfromGodot(t)
        flags = [SEEnum.CLEAR_MOON_CB, SEEnum.CLEAR_MOON_MG, SEEnum.CLEAR_MOON_NN]
        self.assertTrue(res.has(flags).item())
        self.assertFalse(res.above_elev('CB11',10.0).item())
        self.assertFalse(res.above_elev('MG11',10.0).item())
        self.assertTrue(res.above_elev('NN11',10.0).item())
        conds = res.above_elev('NN11',10.0) & res.has(flags)
        self.assertTrue(conds.item())

    def testLogic(self):
        t = Epoch('2026-04-02T17:20:00 TDB')
        godotHandler = GodotHandler(t, t, 30.0, './universe.yml')
        uni = godotHandler.fetch_universe()
        earth = uni.frames.vector3('Moon', 'Earth', 'ICRF', t)
        moon = uni.frames.vector3('Earth', 'Moon', 'ICRF', t)

        for i in range(3):
            self.assertAlmostEqual(-earth[i], moon[i])

    def testTimeSpan(self):
        universe_file = './universe.yml'
        ep1 = Epoch('2026-06-02T00:00:00 TDB')
        ep2 = Epoch('2026-06-02T01:00:00 TDB')
        godotHandler = GodotHandler(ep1, ep2, 30.0, universe_file)

        res = godotHandler.calculate_visibility(200)

        flags = [SEEnum.CLEAR_MOON_NN, SEEnum.SUN_ON_MOON]
        condition = (res.above_elev('CB11', 10.0) & res.has(flags)).values

        old_good = [True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, True, True, True,
                    True, True, True, True, True, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, 
                    False, False, False, False, False]
        for i in range(condition.size):
            self.assertEqual(condition[i], old_good[i])

if __name__ == "__main__":
    unittest.main()