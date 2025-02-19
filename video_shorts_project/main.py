from src.video_shorts_generator import VideoShortsGenerator

def main():
    generator = VideoShortsGenerator(
        video_path="input/your_video.mp4",
        srt_path="input/your_subtitles.srt",
        output_dir="output_shorts"
    )
    generator.generate_shorts(num_segments=100)

if __name__ == "__main__":
    main()
