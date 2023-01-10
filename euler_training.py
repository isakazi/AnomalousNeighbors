from copy import deepcopy
import argparse

import pandas as pd
import torch
from torch.optim import Adam

import pickle

from utils import generators as g
import utils.load_euler as vd
from euler.euler import EulerGCN
from utils.score_utils import get_score

torch.set_num_threads(8)

NUM_TESTS = 1
PATIENCE = 100
MAX_DECREASE = 2
TEST_TS = 1

fmt_score = lambda x : 'AUC: %0.4f AP: %0.4f' % (x[0], x[1])


def train(model, data, output_path, epochs=150, pred=False, nratio=1, lr=0.01):
    print(f'lr:{lr}, epochs: {epochs}')
    end_tr = data.T - TEST_TS

    opt = Adam(model.parameters(), lr=lr)

    best = (0, None)
    no_improvement = 0
    ew_fn = data.tr_attributes

    for e in range(epochs):
        model.train()
        opt.zero_grad()
        zs = None

        # Get embedding
        zs = model(data.x, data.eis, data.tr, ew_fn=ew_fn)[:end_tr]

        if not pred:
            p, n, z = g.link_detection(data, data.tr, zs, nratio=nratio)

        else:
            p, n, z = g.link_prediction(data, data.tr, zs, nratio=nratio)

        loss = model.loss_fn(p, n, z)
        loss.backward()
        opt.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        trloss = loss.item()
        with torch.no_grad():
            model.eval()
            zs = model(data.x, data.eis, data.tr, ew_fn=ew_fn)[:end_tr]

            if not pred:
                p, n, z = g.link_detection(data, data.va, zs)
                st, sf = model.score_fn(p, n, z)
                sscores = get_score(st, sf)

                print(
                    '[%d] Loss: %0.4f  \n\tSt %s ' %
                    (e, trloss, fmt_score(sscores)),
                    end=''
                )

                avg = sscores[0] + sscores[1]

            else:
                dp, dn, dz = g.link_prediction(data, data.va, zs, include_tr=False)
                dt, df = model.score_fn(dp, dn, dz)
                dscores = get_score(dt, df)
                print(
                    '[%d] Loss: %0.4f  \n\tPr  %s ' %
                    (e, trloss, fmt_score(dscores)),
                    end=''
                )

                avg = (
                        dscores[0] + dscores[1]
                )

            if avg > best[0]:
                print('*')
                best = (avg, deepcopy(model))
                no_improvement = 0

            else:
                print()
                if e > 100:
                    no_improvement += 1
                if no_improvement == PATIENCE:
                    print("Early stopping...\n")
                    break

    model = best[1]
    with open(output_path + '/gc_model.pkl', 'wb')as handle:
        pickle.dump(model, handle)
    with torch.no_grad():
        model.eval()

        # Inductive
        if not pred:
            zs = model(data.x, data.eis, data.tr, ew_fn)[end_tr - 1:]

        # Transductive
        else:
            ew_fn = data.all_attributes
            zs = model(data.x, data.eis, data.all, ew_fn)[end_tr - 1:]

        if not pred:
            zs = zs[1:]
            p, n, z = g.link_detection(data, data.te, zs, start=end_tr)
            t, f = model.score_fn(p, n, z)
            sscores = get_score(t, f)

            print(
                '''
                Final scores: 
                    Static LP:  %s
                '''
                % fmt_score(sscores))

            return {'auc': sscores[0], 'ap': sscores[1]}

        else:
            p, n, z = g.link_prediction(data, data.all, zs, start=end_tr - 1)
            t, f = model.score_fn(p, n, z)
            dscores = get_score(t, f)
            nscores = dscores

            print(
                '''
                Final scores: 
                    Dynamic LP:     %s 
                    Dynamic New LP: %s 
                ''' %
                (fmt_score(dscores),
                 fmt_score(nscores))
            )

            return {
                'pred-auc': dscores[0],
                'pred-ap': dscores[1],
                'new-auc': nscores[0],
                'new-ap': nscores[1],
            }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file',
        action='store',
        help='Input file for the training of Euler (list of graph objects, pickle file)'
    )
    parser.add_argument(
        '-n', '--num_nodes_file',
        action='store',
        help='This loads the number of nodes in the training set (cannot be determined from the input because input is'
             'split by days and some nodes may appear in some days but not in others, so this needs to be provided '
             'explicitly)'
    )
    parser.add_argument(
        '-p', '--predict',
        action='store_true',
        help='Sets model to train on link prediction rather than detection'
    )
    parser.add_argument(
        '--lstm',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.02,
        help='Euler learning rate'
    )
    parser.add_argument(
        '-o', '--output',
        action='store',
        help='directory where output is stored'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        action='store',
        default=150,
        help='number of epochs for the training phase'
    )

    args = parser.parse_args()
    outf = 'euler.txt'
    with open(args.num_nodes_file, 'r') as handle:
        num_nodes = int(handle.readline())
        print(f'num nodes is: {num_nodes}')
    l = [args.file]
    for d in l:
        data = vd.load_gc_data(d, num_nodes)
        model = EulerGCN(data.x.size(1), 32, 16, lstm=args.lstm)

        stats = [
            train(
                deepcopy(model),
                data,
                args.output,
                pred=args.predict,
                lr=args.lr,
                epochs=args.num_epochs,
            ) for num_test in range(NUM_TESTS)
        ]

        df = pd.DataFrame(stats)
        print(df.mean() * 100)
        print(df.sem() * 100)

        f = open(outf, 'a')
        f.write(d + '\n')
        f.write('LR: %0.4f\n' % args.lr)
        f.write(str(df.mean() * 100) + '\n')
        f.write(str(df.sem() * 100) + '\n\n')
        f.close()
