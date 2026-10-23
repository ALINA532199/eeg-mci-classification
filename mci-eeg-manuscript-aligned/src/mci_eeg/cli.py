import argparse


def main():
    parser=argparse.ArgumentParser(description="Revised EEG workflow, not historical result reproduction")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("demo");p.add_argument("--output",required=True);p.add_argument("--n",type=int,default=24)
    for command in ("validate","formal","explain"):
        p=sub.add_parser(command)
        p.add_argument("--data",required=True);p.add_argument("--config",required=True);p.add_argument("--output",required=True)
    p=sub.add_parser("ica-review")
    for key in ("manifest","config","output"):
        p.add_argument("--"+key,required=True)
    p=sub.add_parser("prepare")
    p.add_argument("--ica-directory",required=True)
    for key in ("manifest","behavior","config","output"):
        p.add_argument("--"+key,required=True)
    args=parser.parse_args()
    if args.command=="demo":
        from .demo import make_demo
        make_demo(args.output,args.n)
    elif args.command=="ica-review":
        from .preprocess import review_ica
        review_ica(args.manifest,args.config,args.output)
    elif args.command=="prepare":
        from .preprocess import prepare
        prepare(args.manifest,args.behavior,args.config,args.output,args.ica_directory)
    elif args.command=="explain":
        from .explain import explain
        explain(args.data,args.config,args.output)
    else:
        from .workflow import run
        run(args.data,args.config,args.output,args.command)


if __name__=="__main__":
    main()
