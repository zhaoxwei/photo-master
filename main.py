import os
# Fix for OpenMP runtime conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
from image_deduplicator import ImageDeduplicator

def main():
    parser = argparse.ArgumentParser(description='Smart Photo Deduplication Tool')
    parser.add_argument('--folder', default='D:\\phototest',
                      help='Photo directory path (default: D:\\phototest)')
    parser.add_argument('--threshold', type=float, default=0.65,
                      help='Similarity threshold (0.55-0.75)')
    args = parser.parse_args()

    print(f"\nInitializing photo deduplication tool...")
    print(f"Directory path: {args.folder}")
    print(f"Similarity threshold: {args.threshold}")

    deduplicator = ImageDeduplicator(args.folder)
    deduplicator.find_similar_images(similarity_threshold=args.threshold)

if __name__ == '__main__':
    main() 