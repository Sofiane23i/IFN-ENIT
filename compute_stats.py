import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import argparse
from PIL import Image

# parse command-line options for flexibility
parser = argparse.ArgumentParser(description="Compute annotation statistics")
parser.add_argument('--xml_folder', type=str,
                    help='Path to annotation XML folder (Windows or WSL)')
parser.add_argument('--bmp_folder', type=str,
                    help='Path to BMP image folder (Windows or WSL)')
args = parser.parse_args()

# default annotation path(s)
xml_folder = args.xml_folder if args.xml_folder else r"d:\HTR\Dataset\IFN-ENIT\Annotations"
# default bmp path (used for dimension stats)
bmp_folder = args.bmp_folder if args.bmp_folder else r"d:\HTR\Dataset\IFN-ENIT\Bmp files"

# helper to convert Windows-style path to WSL style

def to_wsl_path(path):
    if len(path) >= 2 and path[1] == ':' and path[0].isalpha():
        drive = path[0].lower()
        rest = path[2:].replace('\\', '/')
        return f"/mnt/{drive}{rest}"
    return path

# resolve folder existence, including alternate and WSL translations
def resolve_folder(path, alternates=[]):
    if os.path.isdir(path):
        return path
    candidates = [path] + alternates + [to_wsl_path(path)] + [to_wsl_path(a) for a in alternates]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"None of expected folders exist: {candidates}")

xml_folder = resolve_folder(xml_folder, [r"d:\HTR\Dataset\IFN-ENIT\xml files"])
bmp_folder = resolve_folder(bmp_folder, [])

char_counts = Counter()
pos_counts = defaultdict(Counter)
word_lengths = []
bigram_counts = Counter()
repeated_bigrams = Counter()
total_files = 0
# image dimension statistics
widths = []
heights = []

for fname in os.listdir(xml_folder):
    if not fname.lower().endswith('.xml'):
        continue
    total_files += 1
    path = os.path.join(xml_folder, fname)
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        continue

    # record image dimension if BMP exists
    # attempt to get filename from xml
    file_name = None
    for record in root.findall('record'):
        name_elem = record.find('file_name')
        if name_elem is not None and name_elem.text:
            file_name = name_elem.text.strip()
            break
    if not file_name:
        file_name = os.path.splitext(fname)[0]
    bmp_path = os.path.join(bmp_folder, f"{file_name}.bmp")
    if os.path.isfile(bmp_path):
        try:
            with Image.open(bmp_path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception:
            pass

    items = []
    for obj in root.findall('.//object'):
        name_elem = obj.find('name')
        bbox = obj.find('bndbox')
        if name_elem is None or bbox is None:
            continue
        char = name_elem.text or ''
        try:
            xmin = int(bbox.find('xmin').text)
        except Exception:
            xmin = 0
        items.append((xmin, char))
    items.sort(key=lambda x: x[0])
    word = ''.join(char for _, char in items)
    if len(word) > 0:
        word_lengths.append(len(word))
        for idx, ch in enumerate(word, start=1):
            char_counts[ch] += 1
            pos_counts[idx][ch] += 1
        for a, b in zip(word, word[1:]):
            bigram = a + b
            bigram_counts[bigram] += 1
            if a == b:
                repeated_bigrams[bigram] += 1

max_len = max(word_lengths) if word_lengths else 0
avg_len = sum(word_lengths) / len(word_lengths) if word_lengths else 0

print(f"Processed {total_files} xml files")
print(f"Total words analyzed: {len(word_lengths)}")
print(f"Max word length: {max_len}")
print(f"Average word length: {avg_len:.2f}")

print("\nCharacter occurrences overall:")
for ch, cnt in char_counts.most_common():
    print(f"{ch}: {cnt}")

print("\nCharacter occurrences by position:")
for pos in sorted(pos_counts.keys()):
    print(f"Position {pos}:")
    for ch, cnt in pos_counts[pos].most_common(5):
        print(f"  {ch}: {cnt}")

print("\nTop 20 bigrams:")
for bg, cnt in bigram_counts.most_common(20):
    print(f"{bg}: {cnt}")

print("\nRepeated-character bigrams and their counts:")
for bg, cnt in repeated_bigrams.most_common(20):
    print(f"{bg}: {cnt}")

# --- plotting section ---------------------------------------------------
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not found, skipping plots")
    plt = None

if plt is not None:
    # create output directory for plots
    plot_dir = os.path.join(os.getcwd(), "stats_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # histogram of word lengths
    plt.figure()
    plt.hist(word_lengths, bins=range(1, max_len + 2), edgecolor='black')
    plt.title('Word Length Distribution')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(os.path.join(plot_dir, 'word_length_hist.png'))

    # dimension histograms
    if widths and heights:
        plt.figure()
        plt.hist(widths, bins=50, edgecolor='black')
        plt.title('Image Width Distribution')
        plt.xlabel('Width (pixels)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(plot_dir, 'width_hist.png'))

        plt.figure()
        plt.hist(heights, bins=50, edgecolor='black')
        plt.title('Image Height Distribution')
        plt.xlabel('Height (pixels)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(plot_dir, 'height_hist.png'))

    # bar chart of overall character frequencies (top 30)
    chars, counts = zip(*char_counts.most_common(30))
    plt.figure(figsize=(10,4))
    plt.bar(chars, counts)
    plt.title('Top 30 Character Frequencies')
    plt.xlabel('Character')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'char_freq.png'))

    # line plot of character occurrences by position for top 5 characters
    top_chars = [ch for ch,_ in char_counts.most_common(5)]
    plt.figure(figsize=(8,4))
    for ch in top_chars:
        positions = sorted(pos_counts.keys())
        vals = [pos_counts[p].get(ch,0) for p in positions]
        plt.plot(positions, vals, marker='o', label=ch)
    plt.title('Character Occurrence by Position (Top 5)')
    plt.xlabel('Position in word')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'char_position.png'))

    # bar chart top bigrams
    bgs, bgc = zip(*bigram_counts.most_common(20))
    plt.figure(figsize=(10,4))
    plt.bar(bgs, bgc)
    plt.title('Top 20 Bigrams')
    plt.xlabel('Bigram')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plot_dir, 'bigram_freq.png'))

    # bar chart repeated bigrams
    rbs, rbc = zip(*repeated_bigrams.most_common(20)) if repeated_bigrams else ([],[])
    if rbs:
        plt.figure(figsize=(10,3))
        plt.bar(rbs, rbc)
        plt.title('Repeated-character Bigrams')
        plt.xlabel('Bigram')
        plt.ylabel('Count')
        plt.savefig(os.path.join(plot_dir, 'repeated_bigrams.png'))

    print(f"\nPlots saved in {plot_dir}")
