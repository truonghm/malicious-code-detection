import argparse
import os


def get_all_files(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        all_files.extend([os.path.join(root, file) for file in files])
    return all_files


def move_files_to_root(files, root_path):
    for file in files:
        os.rename(file, os.path.join(root_path, os.path.basename(file)))


def remove_empty_dirs(path):
    for root, dirs, _ in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)


def get_direct_files_count(path):
    return len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Path to the directory")
    args = parser.parse_args()

    # Get the path to "badjs" directory
    dir = args.dir

    # Get the list of all files recursively and print the count
    all_files = get_all_files(dir)
    print(f"Total files recursively inside 'badjs': {len(all_files)}")

    # Move all files to the root of "badjs"
    move_files_to_root(all_files, dir)

    # Delete all empty directories
    remove_empty_dirs(dir)

    # Get the count of all files directly inside "badjs" and print the count
    direct_files_count = get_direct_files_count(dir)
    print(f"Total files directly inside 'badjs': {direct_files_count}")
