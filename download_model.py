from huggingface_hub import snapshot_download

print("Starting download of google/gemma-3-4b-it...")

snapshot_download(
    repo_id="google/gemma-3-4b-it",
    local_dir="gemma-3-4b-it",
    local_dir_use_symlinks=False
)

print("Download complete!")
