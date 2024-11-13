import json
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import youtube_dl


class YouTubeTranscriptSummarizer:
    def __init__(
        self, api_key, model="mixtral-8x7b-32768", temperature=0.0, max_retries=2
    ):
        """
        Initializes the summarizer with the specified parameters.

        Args:
            api_key (str): API key for ChatGroq.
            model (str): Model name to use.
            temperature (float): Sampling temperature.
            max_retries (int): Number of retries for API calls.
        """
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            "Summarize the transcript (maximum 3 sentences): {transcript}"
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def get_transcript(self, video_id, language="en"):
        """
        Fetches the transcript text for a specified YouTube video.

        Args:
            video_id (str): The YouTube video ID.
            language (str): Language code for the transcript.

        Returns:
            str or None: The transcript text or None if unavailable.
        """
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(
                video_id, languages=[language]
            )
            transcript_text = " ".join(
                [
                    entry["text"]
                    for entry in transcript_data
                    if not entry["text"].startswith("[")
                ]
            )
            return transcript_text
        except Exception as e:
            print(f"Error fetching transcript for video ID {video_id}: {e}")
            return None

    def truncate_text(self, text, word_limit=1000):
        """
        Truncates text to a maximum number of words.

        Args:
            text (str): The text to truncate.
            word_limit (int): Maximum number of words allowed.

        Returns:
            str: The truncated text.
        """
        words = text.split()
        if len(words) > word_limit:
            truncated_text = " ".join(words[:word_limit])
            return truncated_text
        return text

    def summarize_transcript(self, video_id):
        """
        Fetches, truncates, and summarizes the transcript.

        Args:
            video_id (str): The YouTube video ID.

        Returns:
            str or None: The summary or None if unsuccessful.
        """
        transcript = self.get_transcript(video_id)
        if transcript:
            truncated_transcript = self.truncate_text(transcript, word_limit=1000)
            try:
                summary = self.chain.invoke(truncated_transcript)
                return summary
            except Exception as e:
                print(f"Error generating summary for video ID {video_id}: {e}")
                return None
        else:
            print(f"Transcript not available for video ID {video_id}.")
            return None


class VideoInfoFetcher:
    def __init__(self, summarizer, output_path="zad-academy_llm.json"):
        """
        Initializes the video info fetcher.

        Args:
            summarizer (YouTubeTranscriptSummarizer): Instance of the summarizer.
            output_path (str): Path to save the output JSON file.
        """
        self.summarizer = summarizer
        self.output_path = output_path

    def fetch_video_info(self, channel_url, reference, max_videos=5):
        """
        Fetches video information from a YouTube channel and summarizes their transcripts.

        Args:
            channel_url (str): URL of the YouTube channel.
            reference (str): Reference name or identifier.
            max_videos (int): Maximum number of videos to process.

        Returns:
            list: A list of dictionaries containing video information.
        """
        ydl_opts = {
            "quiet": True,
            "extract_flat": True,
        }
        video_data = []

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                info_dict = ydl.extract_info(channel_url, download=False)
                entries = info_dict.get("entries", [])
                if not entries:
                    print("No videos found in the provided channel URL.")
                    return video_data

                for video in entries[:max_videos]:
                    video_id = video.get("id")
                    title = video.get("title")
                    if not video_id or not title:
                        print("Incomplete video information; skipping.")
                        continue

                    print(f"Processing Video ID: {video_id}, Title: {title}")
                    content_summary = self.summarizer.summarize_transcript(video_id)
                    video_info = {
                        "title": title,
                        "url": f"https://www.youtube.com/watch?v={video_id}",
                        "content": content_summary if content_summary else title,
                        "reference": reference,
                    }
                    video_data.append(video_info)

            except Exception as e:
                print(
                    f"Error extracting video info from channel URL {channel_url}: {e}"
                )

        # Write the collected data to a JSON file
        try:
            with open(self.output_path, "w") as json_file:
                json.dump(video_data, json_file, indent=4, ensure_ascii=False)
            print(f"Video information with summaries saved to {self.output_path}")
        except Exception as e:
            print(f"Error writing to output file {self.output_path}: {e}")

        return video_data


def main(reference, channel_url, api_key):
    """
    Main function to fetch and summarize YouTube video transcripts.

    Args:
        reference (str): Reference name or identifier.
        channel_url (str): URL of the YouTube channel.
        api_key (str): API key for ChatGroq.
    """
    # Initialize the summarizer
    summarizer = YouTubeTranscriptSummarizer(api_key=api_key)

    # Initialize the video info fetcher
    fetcher = VideoInfoFetcher(summarizer=summarizer)

    # Fetch and summarize video information
    fetcher.fetch_video_info(channel_url=channel_url, reference=reference)


# Example usage
if __name__ == "__main__":
    # Replace the following variables with your actual inputs
    reference_input = "channel_namexxxx"
    channel_url_input = "https://www.youtube.com/@chaneelidxxx/videos"
    api_key_input = "api_key"

    main(
        reference=reference_input, channel_url=channel_url_input, api_key=api_key_input
    )
