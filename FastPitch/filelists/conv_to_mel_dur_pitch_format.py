from argparse import ArgumentParser


def main():

    parser = ArgumentParser()
    parser.add_argument('-o', '--outdir', dest='outdir')
    parser.add_argument('-f', '--file', dest='file')
    parser.add_argument('-p', '--prefix', dest='prefix')

    args = parser.parse_args()

    with open(f"{args.outdir}/{args.prefix}_mel_dur_pitch.txt", "w") as wr:
        with open(f"{args.file}", "r") as f:
            for line in f:
                l = line.split("|")
                path = l[0].split("/")
                format = path[-1].rsplit(".", 1)
                format[-1] = "pt"
                path[-1] = ".".join(format)
                dirs = []
                for dir_name in ["mels", "durations", "pitch_char"]:
                    path = [dir_name, path[-1]]
                    dirs.append("/".join(path))
#                print("|".join(dirs + l[1:]))
#                break
                wr.write("|".join(dirs + l[1:]))


if __name__ == '__main__':
    main()
