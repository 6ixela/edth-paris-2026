import yt_dlp
import os

URLS = [
    "https://www.youtube.com/watch?v=KhNeOZM7dHs",  # ATC JFK
    "https://www.youtube.com/watch?v=2-ASDmJE3Tk"
]

def download_audio(url, output_dir="./data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == "__main__":
    for url in URLS:
        download_audio(url)
        print(f"✅ Téléchargé: {url}")
