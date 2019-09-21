# -*- coding: utf-8 -*-
import logging
import os
import time
import sys



import numpy as np
import pandas as pd







def save_history(export_folder, train_history, x_all, z_all, p_all, w_all, y_all):
    print("Writing path history on disk, {}/".format(export_folder))

    w = np.concatenate(w_all, axis=0)
    x = np.concatenate(x_all, axis=0)
    p = np.concatenate(p_all, axis=0)
    z = np.concatenate(z_all, axis=0)


    np.save(os.path.join(export_folder, 'x.npy'), x)
    # np.save(export_folder + '/y.npy', np.concatenate(y_all, axis=0))
    np.save(os.path.join(export_folder, 'z.npy'), z)
    np.save(os.path.join(export_folder, 'p.npy'), p)
    np.save(export_folder + '/w.npy', w)

    save_stats(export_folder, z,w, x, p)





def save_stats(export_folder, z,w, x, p):
    import pandas as pd
    import matplotlib.pyplot as plt

    #### Weight at ome sample   #############################################
    def get_sample(i):
        dd = {"x1": x[i][0][0][:x.shape[3] - 1],
              "x2": x[i][1][0][:x.shape[3] - 1],
              "x3": x[i][1][0][:x.shape[3] - 1],
              "pret": p[i],

              "z1": z[i][0],
              "z2": z[i][1],
              "z3": z[i][2],

              "w1": w[i][0],
              "w2": w[i][1],
              "w3": w[i][2],


              }

        df = pd.DataFrame(dd)
        return df


    def sample_save(i) :
         try :
           dfw1 = get_sample(i)
           dfw1.to_csv(export_folder + "/weight_sample_{i}.txt".format(i=i) )

           for k in range(0,19) :
               dfw1 = dfw1 + get_sample(i+k)

           dfw1 = dfw1 / 20.0
           dfw1[["w1", "w2", "w3"]].plot()
           plt.savefig(export_folder + "/w_sample_{i}.png".format(i=i) )
           plt.close()
         except :
            pass


    sample_save(  1000  )
    sample_save(  10000 )
    sample_save(  50000 )
    sample_save(  75000 )
    sample_save( 100000 )
    sample_save( 150000 )
    sample_save( 190000 )
    sample_save( 250000 )


    #### Weight Convergence  ###############################################
    dfw = pd.DataFrame(
        {"w" + str(i + 1): w[:, i, -1] for i in range(w.shape[1])}
    )
    dfw.to_csv(export_folder + "/weight_conv.txt")

    dfw.iloc[:, :].plot()
    plt.savefig(export_folder + 'w_conv_all.png')
    plt.close()

    dfw.iloc[:10 ** 5, :].plot()
    plt.savefig(export_folder + 'w_conv_100k.png')
    plt.close()
    # get_sample( 190000 )[ [  "w1", "w2", "w3" ]   ].plot()

    #### Actual Simulation stats : correl, vol  ############################
    ### Sum(return over [0,T])
    dfx = pd.DataFrame({"x" + str(i + 1): np.sum(x[:, i, 0, :], axis=-1)
                        for i in range(x.shape[1])})

    dd = {}
    dd["ww"] = str(list(dfw.iloc[-1, :].values))
    dd["x_vol"] = {k: dfx[k].std() for k in dfx.columns}
    dd["x_corr"] = {}
    from itertools import combinations
    for x1, x2 in list(combinations(dfx.columns, 2)):
        dd["x_corr"][x1 + "_" + x2] = np.corrcoef(dfx[x1].values, dfx[x2].values)[1, 0],

    import json
    json.dump(dd, open(export_folder + "/x_stats2.txt", mode="w"))



