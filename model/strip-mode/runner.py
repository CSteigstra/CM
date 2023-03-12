import json
import random
import sys
import json
import random
import subprocess
from plot_grid import plot
import numpy as np


def solve(lp_file, args, n=0):
    args = ['clingo', '--outf=2', lp_file] + args + ['-n', n, '-t', '2', "--rand-freq=1.0"]
    app = subprocess.Popen(args, stdout=subprocess.PIPE, universal_newlines=True)
    for line in app.stdout:
        yield line.lstrip().rstrip().replace('"', '')

# def solve(args):
#     """Run clingo with the provided argument list and return the parsed JSON result."""
#     fn = args[0]
#     print_args = ['clingo'] + list(args) + [' | tr [:space:] \\\\n | sort ']
#     args = ['clingo', '--outf=2'] + args + ['-t', '2', "--sign-def=rnd","--seed="+str(random.randint(0,1<<30))]
    
#     with subprocess.Popen(
#         ' '.join(args),
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         shell=True
#     ) as clingo:
#         outb, err = clingo.communicate()
#     if err:
#         print(err)
#     out = outb.decode("utf-8")

#     with open(f'{fn}_dump{np.random.randint(0, 200)}.lp', 'w') as outfile:
#         result = json.loads(out)
#         json.dump(result, outfile, indent=4)
    
#     return result

# def main():
#     args = sys.argv[1:]

#     result = solve(args)

#     print(result['Models'])
#     print(result['Time'])

#     if result['Result'] == 'SATISFIABLE':
#         for r in result['Call'][0]['Witnesses']:
#             plot(' '.join(r['Value']))
#     elif result['Result'] == 'OPTIMUM FOUND':
#         plot(' '.join(result['Call'][0]['Witnesses'][-1]['Value']))
#     else:
#         print('No solution found.')

# if __name__ == '__main__':
#     main()
