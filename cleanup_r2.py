"""
Cleanup script to delete old folders from Cloudflare R2
"""
from storage import get_storage

def list_r2_contents(prefix=''):
    """List all files in R2 bucket with optional prefix"""
    storage = get_storage()

    print(f"Fetching files from R2 (prefix: '{prefix}')...")
    all_files = storage.list_files(prefix=prefix)

    # Group by subfolder (two levels deep)
    folders = {}
    for file_key in all_files:
        # Get folder path (e.g., 'jobs/abc123/')
        parts = file_key.split('/')
        if len(parts) >= 2:
            # For jobs/abc123/file.png, folder is 'jobs/abc123'
            folder = '/'.join(parts[:2])
            if folder not in folders:
                folders[folder] = []
            folders[folder].append(file_key)
        elif len(parts) == 1:
            if 'root' not in folders:
                folders['root'] = []
            folders['root'].append(file_key)

    print(f"\nFound {len(all_files)} total files in {len(folders)} folders:")
    print()

    for folder_name in sorted(folders.keys()):
        file_count = len(folders[folder_name])
        print(f"  {folder_name}/  ({file_count} files)")

    return folders

def delete_folder(folder_name, folders):
    """Delete all files in a folder"""
    if folder_name not in folders:
        print(f"Folder '{folder_name}' not found!")
        return

    files = folders[folder_name]
    print(f"\nDeleting {len(files)} files from '{folder_name}/'...")

    storage = get_storage()
    deleted = 0
    failed = 0

    for file_key in files:
        try:
            storage.delete_image(file_key)
            deleted += 1
            if deleted % 10 == 0:
                print(f"  Deleted {deleted}/{len(files)} files...")
        except Exception as e:
            print(f"  Failed to delete {file_key}: {e}")
            failed += 1

    print(f"\n✓ Deleted {deleted} files")
    if failed > 0:
        print(f"✗ Failed to delete {failed} files")

def main():
    print("=" * 60)
    print("Cloudflare R2 Cleanup Script")
    print("=" * 60)
    print()

    # List contents in jobs/ folder
    folders = list_r2_contents(prefix='jobs/')

    print()
    print("Which folder(s) do you want to delete?")
    print("Enter full path like 'jobs/abc123' (or comma-separated list), or 'cancel' to exit:")
    print()

    user_input = input("> ").strip()

    if user_input.lower() == 'cancel':
        print("Cancelled.")
        return

    # Parse input
    folder_names = [f.strip() for f in user_input.split(',')]

    # Confirm deletion
    print()
    print("⚠️  WARNING: This will permanently delete:")
    total_files = 0
    for folder_name in folder_names:
        if folder_name in folders:
            count = len(folders[folder_name])
            total_files += count
            print(f"   - {folder_name}/  ({count} files)")
        else:
            print(f"   - {folder_name}/  (NOT FOUND - will skip)")

    print()
    print(f"Total: {total_files} files will be deleted")
    print()
    confirm = input("Type 'DELETE' to confirm: ").strip()

    if confirm != 'DELETE':
        print("Cancelled.")
        return

    # Delete each folder
    for folder_name in folder_names:
        if folder_name in folders:
            delete_folder(folder_name, folders)

    print()
    print("✓ Cleanup complete!")

if __name__ == '__main__':
    main()
