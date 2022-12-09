import numpy as np
import argparse
from clyngor import ASP
from runner import solve

# def pixel2array(pixel_idx, h, w):
    
#     return r


def main(h, w, n, has_strip):
    program = f'#const w = {w}.\n#const h = {h}.\n'

    with open('gen_strip.lp') as f:
        program += f.read()

    if has_strip:
        program += '\n:- not strip_h(_), not strip_v(_).'
    else:
        program += '\n:- strip_h(_).\n:- strip_v(_).'

    program += '\n#show pixel/1.'

    answers = ASP(program)

    for answer in answers.by_predicate:
        # print(answer['pixel'])
        pix = answer['pixel']
        r = np.zeros(h*w)
        for p in pix:
            x, y = p[0][1]
            
        # r = pixel2array(answer['pixel']
        # pass
        # # if answer.predicate == 'pixel':
        #     pixel_idx = answer.arguments
        #     r = pixel2array(pixel_idx, h, w)
        #     print(r.reshape(h, w))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--height', type=int, default=3)
    parser.add_argument('--width', type=int, default=3)
    parser.add_argument('-n', type=int, default=0)
    parser.add_argument('-s', type=bool, default=True)
    args = parser.parse_args()
    main(args.height, args.width, args.n, args.s)
