from synthesizer.preprocess import create_embeddings
from utils.argutils import print_args
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates embeddings for the synthesizer from the LibriSpeech utterances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("synthesizer_root", type=Path, help=\
        "Path to the synthesizer training data that contains the audios and the train.txt file. "
        "If you let everything as default, it should be <datasets_root>/SV2TTS/synthesizer/.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, 
                        default="encoder/saved_models/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-ne", "--n_processes", type=int, default=4, help= \
        "Number of parallel processes. An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy.")
    parser.add_argument("--use_resnet", action="store_true", help=\
        "Use ResNet encoder instead of LSTM encoder.")
    parser.add_argument("--resnet_model_fpath", type=Path, 
                       default="ResNet/saved_models/resnet_encoder.pt", help=\
        "Path to a saved ResNet encoder model (可使用标准ResNet预训练模型或通过encoder_train_resnet.py训练的模型).")
    args = parser.parse_args()
    
    # Process the arguments
    encoder_model_fpath = args.resnet_model_fpath if args.use_resnet else args.encoder_model_fpath
    
    # Run the preprocessing
    print_args(args, parser)
    create_embeddings(args.synthesizer_root, encoder_model_fpath, args.n_processes, args.use_resnet)
