from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir', dest='outdir')
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-p', '--prefix', dest='prefix')

    args = parser.parse_args()

    with open(f"{args.outdir}/{args.prefix}_fix_paths.txt", "w") as wr:
        with open(f"{args.file}", "r") as f:
            for line in f:
                l = line.split("|")
                filename = 'wavs/'+str(Path(l[0]).stem)+'.wav'
                wr.write("|".join([filename]+line.split("|")[1:]))


if __name__ == '__main__':
    main()
