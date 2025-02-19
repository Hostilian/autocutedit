from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import pysrt
from pathlib import Path
import os
from datetime import timedelta
import random
import math
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
import colorsys

class VideoShortsGenerator:
    def __init__(self, video_path, srt_path, output_dir, min_duration=15, max_duration=60):
        """
        Initialize the video shorts generator with enhanced features
        """
        self.video_path = video_path
        self.srt_path = srt_path
        self.output_dir = Path(output_dir)
        self.min_duration = min_duration
        self.max_duration = max_duration

        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load video and subtitles
        self.video = VideoFileClip(video_path)
        self.subtitles = pysrt.open(srt_path)

        # Initialize subtitle styling
        self.subtitle_styles = self._initialize_subtitle_styles()

    def _initialize_subtitle_styles(self):
        """Initialize a variety of subtitle styles"""
        return {
            'default': {
                'font': 'Arial',
                'fontsize': 40,
                'color': 'white',
                'bg_color': 'black',
                'stroke_color': None,
                'stroke_width': 1,
            },
            'bold': {
                'font': 'Arial-Bold',
                'fontsize': 45,
                'color': 'white',
                'bg_color': 'black',
                'stroke_color': 'white',
                'stroke_width': 2,
            },
            'neon': {
                'font': 'Arial',
                'fontsize': 40,
                'color': '#00ff00',
                'bg_color': None,
                'stroke_color': '#003300',
                'stroke_width': 3,
            },
            'dramatic': {
                'font': 'Arial-Bold',
                'fontsize': 50,
                'color': 'red',
                'bg_color': None,
                'stroke_color': 'black',
                'stroke_width': 3,
            }
        }

    def generate_title(self, subtitle_text, context_subtitles=[]):
        """
        Generate a more sophisticated title based on subtitle content and context
        """
        # Clean and combine all relevant text
        all_text = ' '.join([subtitle_text] + context_subtitles)
        all_text = re.sub(r'[^\w\s]', '', all_text.lower())

        # Tokenize and remove stopwords
        words = word_tokenize(all_text)
        words = [word for word in words if word not in self.stop_words]

        # Get most common meaningful words
        word_freq = Counter(words)
        important_words = [word for word, freq in word_freq.most_common(5) if len(word) > 3]

        # Generate title strategies
        titles = []

        # Strategy 1: Use first sentence if it's concise
        first_sent = sent_tokenize(subtitle_text)[0]
        if len(first_sent.split()) <= 8:
            titles.append(first_sent)

        # Strategy 2: Important words combination
        if important_words:
            titles.append(' '.join(important_words[:3]).title())

        # Strategy 3: Pattern matching for questions
        if '?' in subtitle_text:
            question = [s for s in sent_tokenize(subtitle_text) if '?' in s][0]
            titles.append(question)

        # Choose the best title
        final_title = max(titles, key=lambda x: self._score_title(x))
        return final_title

    def _score_title(self, title):
        """Score a potential title based on various factors"""
        score = 0
        words = title.split()

        # Length preference (5-8 words is ideal)
        if 5 <= len(words) <= 8:
            score += 3
        elif 3 <= len(words) <= 10:
            score += 1

        # Question titles are engaging
        if '?' in title:
            score += 2

        # Prefer titles with numbers
        if any(char.isdigit() for char in title):
            score += 1

        return score

    def get_dynamic_color(self, text_content, frame_time):
        """Generate dynamic colors based on content and timing"""
        # Emotional words for color association
        emotional_colors = {
            'happy': '#FFD700',  # Gold
            'sad': '#4169E1',    # Royal Blue
            'angry': '#FF4500',  # Red-Orange
            'neutral': '#FFFFFF' # White
        }

        # Simple emotion detection (can be enhanced with NLP)
        emotion = 'neutral'
        text_lower = text_content.lower()

        if any(word in text_lower for word in ['happy', 'joy', 'laugh', 'smile']):
            emotion = 'happy'
        elif any(word in text_lower for word in ['sad', 'sorry', 'unfortunate']):
            emotion = 'sad'
        elif any(word in text_lower for word in ['angry', 'mad', 'fury']):
            emotion = 'angry'

        return emotional_colors[emotion]

    def create_subtitle_clip(self, text, duration, style_name='default', frame_time=0):
        """Create an enhanced TextClip for subtitles with dynamic styling"""
        style = self.subtitle_styles[style_name].copy()

        # Apply dynamic color based on content
        if style_name == 'default':
            style['color'] = self.get_dynamic_color(text, frame_time)

        # Create base TextClip
        txt_clip = TextClip(
            text,
            font=style['font'],
            fontsize=style['fontsize'],
            color=style['color'],
            bg_color=style['bg_color'],
            size=(self.video.w * 0.8, None),
            method='caption'
        )

        # Add stroke if specified
        if style['stroke_color'] and style['stroke_width']:
            txt_clip = TextClip(
                text,
                font=style['font'],
                fontsize=style['fontsize'],
                color=style['stroke_color'],
                bg_color=style['bg_color'],
                size=(self.video.w * 0.8, None),
                method='caption'
            ).set_duration(duration)

            main_txt = TextClip(
                text,
                font=style['font'],
                fontsize=style['fontsize'],
                color=style['color'],
                bg_color=style['bg_color'],
                size=(self.video.w * 0.8, None),
                method='caption'
            ).set_duration(duration)

            txt_clip = CompositeVideoClip([txt_clip, main_txt])

        return txt_clip.set_duration(duration)

    def process_segment(self, start_time, end_time, subtitle_texts, output_path):
        """Process a single video segment with enhanced subtitles"""
        video_segment = self.video.subclip(start_time, end_time)
        subtitle_clips = []

        for text, (start, end) in subtitle_texts:
            rel_start = max(0, start - start_time)
            rel_end = min(end_time - start_time, end - start_time)

            if rel_start < rel_end:
                # Choose style based on content
                style_name = 'default'
                if '!' in text:
                    style_name = 'dramatic'
                elif '?' in text:
                    style_name = 'neon'
                elif text.isupper():
                    style_name = 'bold'

                subtitle_clip = self.create_subtitle_clip(
                    text,
                    rel_end - rel_start,
                    style_name,
                    rel_start
                ).set_start(rel_start)

                # Add animation
                if len(text) > 30:  # For longer texts
                    subtitle_clip = subtitle_clip.set_position(('center', 'bottom'))
                else:  # For shorter texts, add some movement
                    subtitle_clip = subtitle_clip.set_position(
                        lambda t: ('center', 'bottom' if t < rel_end - rel_start - 0.5 else 'center')
                    )

                subtitle_clips.append(subtitle_clip)

        final_clip = CompositeVideoClip([video_segment] + subtitle_clips)
        final_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac'
        )

    def generate_shorts(self, num_segments=100):
        """Generate multiple short videos with enhanced features"""
        video_duration = self.video.duration
        avg_duration = min(
            self.max_duration,
            max(self.min_duration, video_duration / num_segments)
        )

        for i in range(num_segments):
            start_time = i * avg_duration
            if start_time >= video_duration:
                break

            end_time = min(video_duration, start_time + avg_duration)

            # Get relevant subtitles with context
            segment_subs = []
            context_subtitles = []

            for sub in self.subtitles:
                sub_start = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds
                sub_end = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds

                if sub_start <= end_time and sub_end >= start_time:
                    segment_subs.append((sub.text, (sub_start, sub_end)))
                    context_subtitles.append(sub.text)

            if not segment_subs:
                continue

            # Generate enhanced title
            title = self.generate_title(
                segment_subs[0][0],
                context_subtitles[1:3] if len(context_subtitles) > 1 else []
            )

            output_path = self.output_dir / f"short_{i+1:03d}_{title[:30]}.mp4"
            self.process_segment(start_time, end_time, segment_subs, output_path)

        self.video.close()

# Example usage
if __name__ == "__main__":
    generator = VideoShortsGenerator(
        video_path="input_video.mp4",
        srt_path="subtitles.srt",
        output_dir="output_shorts"
    )
    generator.generate_shorts(num_segments=100)
