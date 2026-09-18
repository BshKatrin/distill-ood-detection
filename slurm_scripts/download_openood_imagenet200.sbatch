#!/usr/bin/env bash
#SBATCH --job-name=openood-download
#SBATCH --partition=hard
#SBATCH --exclude=daft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

openood_root="${1:-/home/bogush/openood}"
download_only="${OPENOOD_DOWNLOAD_ONLY:-0}"
archive_dir="${openood_root}/archives"
data_dir="${openood_root}/data"
result_dir="${openood_root}/results"
marker_dir="${openood_root}/.extracted"
log_dir="${openood_root}/logs"
mkdir -p "${archive_dir}" "${data_dir}/images_largescale" "${result_dir}" "${marker_dir}" "${log_dir}"
exec >"${log_dir}/download-${SLURM_JOB_ID:-manual}.out" 2>&1

download_drive_file() {
    local drive_id="$1"
    local filename="$2"
    local expected_bytes="$3"
    local destination="${archive_dir}/${filename}"
    local current_bytes=0
    if [[ -f "${destination}" ]]; then
        current_bytes=$(stat --format='%s' "${destination}")
    fi
    if [[ "${current_bytes}" == "${expected_bytes}" ]]; then
        echo "Already downloaded: ${filename}"
        return
    fi
    curl --fail --location --retry 5 --retry-delay 10 --continue-at - \
        --output "${destination}.part" \
        "https://drive.usercontent.google.com/download?id=${drive_id}&export=download&confirm=t"
    current_bytes=$(stat --format='%s' "${destination}.part")
    if [[ "${current_bytes}" != "${expected_bytes}" ]]; then
        echo "Unexpected size for ${filename}: ${current_bytes}; expected ${expected_bytes}" >&2
        exit 1
    fi
    mv "${destination}.part" "${destination}"
}

extract_archive() {
    local filename="$1"
    local destination="$2"
    local marker="${marker_dir}/${filename}.done"
    if [[ "${download_only}" == "1" ]]; then
        return
    fi
    if [[ -f "${marker}" ]]; then
        echo "Already extracted: ${filename}"
        return
    fi
    unzip -tq "${archive_dir}/${filename}"
    unzip -q -n "${archive_dir}/${filename}" -d "${destination}"
    touch "${marker}"
}

download_pids=()
download_drive_file 1lI1j0_fDDvjIt9JlWAw09X8ks-yrR_H1 benchmark_imglist.zip 27962573 &
download_pids+=("$!")
download_drive_file 1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs imagenet200_res18_v1.5.zip 377361239 &
download_pids+=("$!")
download_drive_file 1PzkA-WGG8Z18h0ooL_pDdz9cO-DCIouE ssb_hard.zip 1055841329 &
download_pids+=("$!")
download_drive_file 1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6 ninco.zip 674439979 &
download_pids+=("$!")
download_drive_file 1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj inaturalist.zip 3952024451 &
download_pids+=("$!")
download_drive_file 1VUFXnB_z70uHfdgJG2E_pjYOcEgqM7tE openimage_o.zip 619605926 &
download_pids+=("$!")
download_failed=0
for download_pid in "${download_pids[@]}"; do
    wait "${download_pid}" || download_failed=1
done
if [[ "${download_failed}" == "1" ]]; then
    echo "At least one OpenOOD archive download failed" >&2
    exit 1
fi

extract_archive benchmark_imglist.zip "${data_dir}"
extract_archive imagenet200_res18_v1.5.zip "${result_dir}"
extract_archive ssb_hard.zip "${data_dir}/images_largescale/ssb_hard"
extract_archive ninco.zip "${data_dir}/images_largescale/ninco"
extract_archive inaturalist.zip "${data_dir}/images_largescale/inaturalist"
extract_archive openimage_o.zip "${data_dir}/images_largescale/openimage_o"

if [[ "${download_only}" == "1" ]]; then
    echo "OpenOOD ImageNet-200 archives are downloaded under ${archive_dir}"
else
    echo "OpenOOD ImageNet-200 assets are ready under ${openood_root}"
fi
