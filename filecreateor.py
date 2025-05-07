import os
import sys
import pickle

from mani import GodotHandler

import godot
from godot.core import tempo, util
util.suppressLogger()

if __name__ == '__main__':
    args = sys.argv
    print(args)
    yearbegin = int(args[1])
    yearend = yearbegin + 1

    print("Year: " + str(yearbegin))
    os.makedirs('./output/year_sim/',exist_ok = True)

    # create the universe
    print("Loading universe")
    ep1 = tempo.Epoch(str(yearbegin)+'-01-01T00:00:00 TT')
    ep2 = tempo.Epoch(str(yearend)+'-01-01T00:00:00 TT')

    print("Creating handler, calculating visibility")
    godotHandler = GodotHandler(ep1, ep2, 60.0, './universe2.yml')
    res = godotHandler.calculate_visibility()

    filename = './output/year_sim/one_year_' + str(yearbegin) + '.pickle'

    print("Saving to file")
    with open(filename, 'wb') as f:
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    print("Done saving")