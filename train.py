from allennlp.commands import main
import sys

if __name__ == '__main__':
    sys.argv = [
        "allennlp",  # command name, not used by main
        "train",
        ".\\configs\\drop\\roberta\\drop_robert.jsonnet",
        "-s", "drop_checkpoints",
        "-f",
        "--include-package", "src"
    ]
    main()
