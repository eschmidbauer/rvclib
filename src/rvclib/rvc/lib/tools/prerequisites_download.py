import os
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

file_base = os.environ.get("RVC_MODEL_BASE_PATH", "models")
url_base = "https://huggingface.co/IAHispano/Applio/resolve/main/Resources"

pretraineds_v1_list = [
    (
        "pretrained_v1/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
pretraineds_v2_list = [
    (
        "pretrained_v2/",
        [
            "D32k.pth",
            "D40k.pth",
            "D48k.pth",
            "G32k.pth",
            "G40k.pth",
            "G48k.pth",
            "f0D32k.pth",
            "f0D40k.pth",
            "f0D48k.pth",
            "f0G32k.pth",
            "f0G40k.pth",
            "f0G48k.pth",
        ],
    )
]
models_list = [("predictors/", ["rmvpe.pt", "fcpe.pt"])]
embedders_list = [("embedders/contentvec/", ["pytorch_model.bin", "config.json"])]


folder_mapping_list = {
    "pretrained_v1/": f"{file_base}/rvc/models/pretraineds/pretrained_v1/",
    "pretrained_v2/": f"{file_base}/rvc/models/pretraineds/pretrained_v2/",
    "embedders/contentvec/": f"{file_base}/rvc/models/embedders/contentvec/",
    "predictors/": f"{file_base}/rvc/models/predictors/",
    "formant/": f"{file_base}/rvc/models/formant/",
}


def get_file_size_if_missing(file_list):
    """
    Calculate the total size of files to be downloaded only if they do not exist locally.
    """
    total_size = 0
    for remote_folder, files in file_list:
        local_folder = folder_mapping_list.get(remote_folder, "")
        for file in files:
            destination_path = os.path.join(local_folder, file)
            if not os.path.exists(destination_path):
                url = f"{url_base}/{remote_folder}{file}"
                response = requests.head(url)
                total_size += int(response.headers.get("content-length", 0))
    return total_size


def download_file(url, destination_path, global_bar):
    """
    Download a file from the given URL to the specified destination path,
    updating the global progress bar as data is downloaded.
    """

    dir_name = os.path.dirname(destination_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            global_bar.update(len(data))


def download_mapping_files(file_mapping_list, global_bar):
    """
    Download all files in the provided file mapping list using a thread pool executor,
    and update the global progress bar as downloads progress.
    """
    with ThreadPoolExecutor() as executor:
        futures = []
        for remote_folder, file_list in file_mapping_list:
            local_folder = folder_mapping_list.get(remote_folder, "")
            for file in file_list:
                destination_path = os.path.join(local_folder, file)
                if not os.path.exists(destination_path):
                    url = f"{url_base}/{remote_folder}{file}"
                    futures.append(executor.submit(download_file, url, destination_path, global_bar))
        for future in futures:
            future.result()


def split_pretraineds(pretrained_list):
    f0_list = []
    non_f0_list = []
    for folder, files in pretrained_list:
        f0_files = [f for f in files if f.startswith("f0")]
        non_f0_files = [f for f in files if not f.startswith("f0")]
        if f0_files:
            f0_list.append((folder, f0_files))
        if non_f0_files:
            non_f0_list.append((folder, non_f0_files))
    return f0_list, non_f0_list


pretraineds_v1_f0_list, pretraineds_v1_nof0_list = split_pretraineds(pretraineds_v1_list)
pretraineds_v2_f0_list, pretraineds_v2_nof0_list = split_pretraineds(pretraineds_v2_list)


def calculate_total_size(
    pretraineds_v1_f0,
    pretraineds_v1_nof0,
    pretraineds_v2_f0,
    pretraineds_v2_nof0,
    models_list,
    embedders_list,
):
    """
    Calculate the total size of all files to be downloaded based on selected categories.
    """
    total_size = 0
    total_size += get_file_size_if_missing(pretraineds_v1_f0)
    total_size += get_file_size_if_missing(pretraineds_v1_nof0)
    total_size += get_file_size_if_missing(pretraineds_v2_f0)
    total_size += get_file_size_if_missing(pretraineds_v2_nof0)
    total_size += get_file_size_if_missing(models_list)
    total_size += get_file_size_if_missing(embedders_list)
    return total_size


def prequisites_download_pipeline():
    """
    Manage the download pipeline for different categories of files.
    """
    total_size = calculate_total_size(
        pretraineds_v1_f0_list,
        pretraineds_v1_nof0_list,
        pretraineds_v2_f0_list,
        pretraineds_v2_nof0_list,
        models_list,
        embedders_list
    )

    if total_size > 0:
        with tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading files") as global_bar:
            download_mapping_files(pretraineds_v1_f0_list, global_bar)
            download_mapping_files(pretraineds_v1_nof0_list, global_bar)
            download_mapping_files(pretraineds_v2_f0_list, global_bar)
            download_mapping_files(pretraineds_v2_nof0_list, global_bar)
            download_mapping_files(models_list, global_bar)
            download_mapping_files(embedders_list, global_bar)
    else:
        pass


if __name__ == "__main__":
    prequisites_download_pipeline()
