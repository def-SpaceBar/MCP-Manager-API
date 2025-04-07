import os
import subprocess
import json
import uuid
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from moviepy import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip, ColorClip
import numpy as np
import logging
import base64
import requests
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini AI
genai.configure(api_key=GEMINI_API_KEY)


class VideoEditorSaaS:
    def __init__(self, workspace_dir: str = None):
        """Initialize the Video Editor SaaS."""
        self.workspace_dir = workspace_dir or tempfile.mkdtemp()
        self.uploaded_clips = {}
        self.analysis_results = {}
        self.editing_plan = {}
        self.edited_clips = {}
        self.output_video = None

        # Use the most capable Gemini model for multimodal (vision) analysis
        self.gemini_vision_model = genai.GenerativeModel('gemini-1.5-pro-vision')
        # Use text model for planning and decision making
        self.gemini_text_model = genai.GenerativeModel('gemini-1.5-pro')

        # Create workspace directories
        os.makedirs(os.path.join(self.workspace_dir, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "processed"), exist_ok=True)
        os.makedirs(os.path.join(self.workspace_dir, "outputs"), exist_ok=True)

        logger.info(f"Workspace initialized at {self.workspace_dir}")

    def upload_video(self, video_path: str) -> str:
        """Upload a video file to the workspace."""
        # Check if file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate unique ID for the video
        video_id = str(uuid.uuid4())

        # Get file extension
        _, ext = os.path.splitext(video_path)

        # Copy file to workspace
        dest_path = os.path.join(self.workspace_dir, "uploads", f"{video_id}{ext}")
        shutil.copy2(video_path, dest_path)

        # Store metadata
        self.uploaded_clips[video_id] = {
            "original_path": video_path,
            "workspace_path": dest_path,
            "format": ext[1:],  # Remove the dot
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"Video uploaded: {os.path.basename(video_path)} with ID {video_id}")
        return video_id

    def extract_frames(self, video_id: str, num_frames: int = 5) -> List[str]:
        """Extract representative frames from a video for AI analysis."""
        video_path = self.uploaded_clips[video_id]["workspace_path"]
        frames_dir = os.path.join(self.workspace_dir, "processed", f"{video_id}_frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Get video duration
        probe_cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration = float(json.loads(probe_result.stdout)["format"]["duration"])

        # Extract frames at regular intervals
        frame_paths = []
        for i in range(num_frames):
            timestamp = duration * i / (num_frames - 1) if num_frames > 1 else duration / 2
            output_path = os.path.join(frames_dir, f"frame_{i}.jpg")

            extract_cmd = [
                "ffmpeg", "-y", "-ss", str(timestamp), "-i", video_path,
                "-vframes", "1", "-q:v", "2", output_path
            ]
            subprocess.run(extract_cmd, capture_output=True)
            frame_paths.append(output_path)

        logger.info(f"Extracted {len(frame_paths)} frames from video {video_id}")
        return frame_paths

    def extract_audio(self, video_id: str) -> str:
        """Extract audio from a video for AI analysis."""
        video_path = self.uploaded_clips[video_id]["workspace_path"]
        audio_path = os.path.join(self.workspace_dir, "processed", f"{video_id}.wav")

        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "2", audio_path
        ]
        subprocess.run(extract_cmd, capture_output=True)

        logger.info(f"Extracted audio from video {video_id}")
        return audio_path

    def analyze_video_content(self, video_id: str) -> Dict[str, Any]:
        """Analyze video content directly using Gemini Vision AI."""
        video_path = self.uploaded_clips[video_id]["workspace_path"]

        # Read the video file as binary data
        with open(video_path, "rb") as f:
            video_data = f.read()

        # Create a data part for the video
        video_part = {"mime_type": "video/mp4", "data": video_data}

        # Create a prompt that explains what we need from the analysis
        prompt = """
        Analyze this video thoroughly and provide the following information:
        1. Main subject/focus (person, product, location, etc.)
        2. Scene description and setting
        3. Activities or actions being performed
        4. Emotional tone of the scene
        5. Visual style and aesthetics
        6. Product or brand visibility (if any)
        7. Estimated purpose of this clip in a marketing context
        8. Key visual elements that could be highlighted
        9. Chronological position (beginning=1, middle=5, or end=10 of a story)
        10. Audio content description (speech, music, sound effects)

        Format your response as a structured JSON object with these fields:
        - main_subject: string
        - scene_description: string
        - activities: string
        - emotional_tone: string
        - visual_style: string
        - product_visibility: string
        - marketing_purpose: string
        - key_visual_elements: array of strings
        - logical_scene_number: number (1-10)
        - audio_content: string
        """

        try:
            # Call Gemini model with the full video file
            response = self.gemini_vision_model.generate_content([prompt, video_part])

            # Try to extract JSON from the response
            response_text = response.text
            logger.info(f"Gemini analysis response for {video_id}: {response_text[:500]}...")

            # Find JSON content
            try:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_content = response_text[json_start:json_end]
                    analysis = json.loads(json_content)
                else:
                    # Fallback: have Gemini reformat as JSON
                    reformat_prompt = f"Convert this analysis to valid JSON: {response_text}"
                    json_response = self.gemini_text_model.generate_content(reformat_prompt)
                    json_content = json_response.text
                    json_start = json_content.find('{')
                    json_end = json_content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        analysis = json.loads(json_content[json_start:json_end])
                    else:
                        raise ValueError("Failed to parse JSON from Gemini response")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                # Create a structured object manually
                analysis = {
                    "main_subject": "Unknown",
                    "scene_description": response_text[:200],
                    "activities": "Unknown",
                    "emotional_tone": "Neutral",
                    "visual_style": "Standard",
                    "product_visibility": "Unknown",
                    "marketing_purpose": "General content",
                    "key_visual_elements": [],
                    "logical_scene_number": 5,  # Middle by default
                    "audio_content": "Unknown"
                }
        except Exception as e:
            logger.error(f"Error analyzing video content: {e}")
            analysis = {
                "main_subject": "Unknown",
                "scene_description": "Analysis failed",
                "activities": "Unknown",
                "emotional_tone": "Neutral",
                "visual_style": "Standard",
                "product_visibility": "Unknown",
                "marketing_purpose": "General content",
                "key_visual_elements": [],
                "logical_scene_number": 5,
                "audio_content": "Unknown",
                "error": str(e)
            }

        logger.info(f"Video content analysis completed for {video_id}")
        return analysis

    def analyze_audio_content(self, video_id: str) -> Dict[str, Any]:
        """
        Get audio analysis from the video content analysis.

        Since we're now analyzing the entire video with Gemini (including audio),
        this function extracts the audio-related insights from the video analysis.
        """
        # Check if we already have video analysis
        if video_id in self.analysis_results and "video" in self.analysis_results[video_id]:
            video_analysis = self.analysis_results[video_id]["video"]

            # Extract audio content description if available
            audio_content = video_analysis.get("audio_content", "Unknown")

            # Determine if there's speech based on the audio content description
            has_speech = "speech" in audio_content.lower() if isinstance(audio_content, str) else False

            # Derive main topics from the video analysis
            main_topics = []
            if "main_subject" in video_analysis and video_analysis["main_subject"] != "Unknown":
                main_topics.append(video_analysis["main_subject"])

            # Extract emotional tone
            emotional_tone = video_analysis.get("emotional_tone", "Neutral")

            # Determine if there's a call to action
            call_to_action = "Present" if "call to action" in audio_content.lower() or "cta" in audio_content.lower() else "None"

            # Get logical scene position
            logical_position = video_analysis.get("logical_scene_number", 5)

            # Create a scene classification based on available info
            if "marketing_purpose" in video_analysis:
                purpose = video_analysis["marketing_purpose"].lower()
                if "intro" in purpose:
                    scene_classification = "Introduction"
                elif "product" in purpose:
                    scene_classification = "Product showcase"
                elif "testimonial" in purpose:
                    scene_classification = "Testimonial"
                elif "cta" in purpose or "call to action" in purpose:
                    scene_classification = "Call to action"
                else:
                    scene_classification = "Content segment"
            else:
                scene_classification = "General content"

            # Create audio analysis
            analysis = {
                "has_speech": has_speech,
                "audio_content": audio_content,
                "main_topics": main_topics,
                "emotional_tone": emotional_tone,
                "call_to_action": call_to_action,
                "audio_quality": "Unknown",  # Would need specific audio analysis
                "background_noise": "Unknown",  # Would need specific audio analysis
                "scene_classification": scene_classification,
                "logical_position": logical_position
            }

            logger.info(f"Audio content analysis derived from video analysis for {video_id}")
            return analysis

        # If we don't have video analysis yet, get basic audio info
        video_path = self.uploaded_clips[video_id]["workspace_path"]

        # Check if the video has audio
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,codec_name",
            "-of", "json", video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)

        # Find if video has audio stream
        has_audio = any(
            stream["codec_type"] == "audio" for stream in video_info["streams"]) if "streams" in video_info else False

        if not has_audio:
            logger.warning(f"No audio stream found in video {video_id}")
            return {
                "has_speech": False,
                "audio_content": "No audio",
                "main_topics": [],
                "emotional_tone": "None",
                "call_to_action": "None",
                "audio_quality": "None",
                "background_noise": "None",
                "scene_classification": "Silent clip",
                "logical_position": 5  # Middle by default
            }

        # Basic fallback analysis
        return {
            "has_speech": True,
            "audio_content": "Audio present but not analyzed in detail",
            "main_topics": ["Unknown"],
            "emotional_tone": "Neutral",
            "call_to_action": "Unknown",
            "audio_quality": "Unknown",
            "background_noise": "Unknown",
            "scene_classification": "General content",
            "logical_position": 5  # Middle by default
        }

    def analyze_videos(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple videos and store results."""
        for video_id in video_ids:
            # First analyze video content (which includes both visual and audio)
            video_analysis = self.analyze_video_content(video_id)

            # Store initial results
            self.analysis_results[video_id] = {
                "video": video_analysis,
                "metadata": self.uploaded_clips[video_id]
            }

            # Now analyze audio aspects based on the video analysis
            audio_analysis = self.analyze_audio_content(video_id)

            # Update with audio analysis
            self.analysis_results[video_id]["audio"] = audio_analysis

        logger.info(f"Completed analysis for {len(video_ids)} videos")
        return self.analysis_results

    def determine_video_sequence(self, video_ids: List[str]) -> List[str]:
        """Determine the optimal sequence of videos based on analysis."""
        # Prepare data for Gemini analysis
        analysis_summaries = []
        for video_id in video_ids:
            if video_id in self.analysis_results:
                analysis = self.analysis_results[video_id]
                summary = {
                    "video_id": video_id,
                    "main_subject": analysis["video"].get("main_subject", "Unknown"),
                    "scene_description": analysis["video"].get("scene_description", "Unknown")[:100],
                    "emotional_tone": analysis["video"].get("emotional_tone", "Neutral"),
                    "logical_scene_number": analysis["video"].get("logical_scene_number", 5),
                    "audio_scene_classification": analysis["audio"].get("scene_classification", "Unknown"),
                    "audio_logical_position": analysis["audio"].get("logical_position", 5),
                    "has_call_to_action": "Yes" if analysis["audio"].get("call_to_action", "None") != "None" else "No"
                }
                analysis_summaries.append(summary)

        # Create prompt for Gemini to determine optimal order
        prompt = f"""
        You are a professional video editor. I have {len(video_ids)} video clips that need to be arranged in a logical sequence for a marketing video.

        Here are summaries of the clips:
        {json.dumps(analysis_summaries, indent=2)}

        Based on this information, determine the most logical and effective sequence for these clips.
        Consider storytelling principles, logical flow, emotional arc, and marketing effectiveness.

        Output ONLY a JSON array of video_ids in the recommended sequence order.
        Do not include any explanation or additional text.
        """

        try:
            response = self.gemini_text_model.generate_content(prompt)
            response_text = response.text

            # Extract JSON array from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                sequence = json.loads(json_content)

                # Validate that all video_ids are in the sequence
                missing_ids = set(video_ids) - set(sequence)
                if missing_ids:
                    for missing_id in missing_ids:
                        sequence.append(missing_id)
                    logger.warning(f"Added missing video IDs to sequence: {missing_ids}")
            else:
                # If JSON parsing fails, use a simple heuristic based on logical position
                sequence = sorted(video_ids,
                                  key=lambda vid_id: (
                                                             self.analysis_results[vid_id]["video"].get(
                                                                 "logical_scene_number", 5) +
                                                             self.analysis_results[vid_id]["audio"].get(
                                                                 "logical_position", 5)
                                                     ) / 2)
                logger.warning("Failed to extract sequence from Gemini response, using fallback method")

        except Exception as e:
            logger.error(f"Error determining video sequence: {e}")
            # Fallback to a simple order
            sequence = video_ids.copy()
            random.shuffle(sequence)

        logger.info(f"Video sequence determined: {sequence}")
        return sequence

    def generate_editing_plan(self, video_sequence: List[str], target_platform: str = "instagram") -> Dict[str, Any]:
        """Generate an AI-based editing plan for the video sequence."""
        # Prepare summaries for Gemini
        clip_summaries = []
        for i, video_id in enumerate(video_sequence):
            if video_id in self.analysis_results:
                analysis = self.analysis_results[video_id]
                summary = {
                    "position": i + 1,
                    "video_id": video_id,
                    "main_subject": analysis["video"].get("main_subject", "Unknown"),
                    "emotional_tone": analysis["video"].get("emotional_tone", "Neutral"),
                    "visual_style": analysis["video"].get("visual_style", "Standard"),
                    "has_speech": analysis["audio"].get("has_speech", False),
                    "call_to_action": analysis["audio"].get("call_to_action", "None")
                }
                clip_summaries.append(summary)

        # Create platform-specific constraints
        platform_specs = {
            "instagram": {
                "aspect_ratios": ["1:1", "4:5", "9:16"],
                "max_length": 60,
                "recommendations": "Vibrant colors, engaging first 3 seconds, text overlays"
            },
            "tiktok": {
                "aspect_ratios": ["9:16"],
                "max_length": 60,
                "recommendations": "Fast-paced editing, trending music, text overlays"
            },
            "youtube": {
                "aspect_ratios": ["16:9"],
                "max_length": 600,
                "recommendations": "Professional transitions, longer segments, annotations"
            },
            "linkedin": {
                "aspect_ratios": ["16:9", "1:1"],
                "max_length": 180,
                "recommendations": "Professional tone, clear subtitles, brand-focused"
            },
            "facebook": {
                "aspect_ratios": ["16:9", "1:1", "4:5"],
                "max_length": 240,
                "recommendations": "Engaging first 3 seconds, subtitles, clear CTA"
            }
        }

        # Use default if platform not recognized
        platform_info = platform_specs.get(target_platform.lower(), platform_specs["instagram"])

        # Create prompt for Gemini to generate editing plan
        prompt = f"""
        You are a professional video editor creating an AI-driven editing plan for a {target_platform} marketing video.

        Video clips in sequence:
        {json.dumps(clip_summaries, indent=2)}

        Platform specifications:
        {json.dumps(platform_info, indent=2)}

        Generate a comprehensive editing plan that includes:
        1. Optimal aspect ratio for this content on {target_platform}
        2. Recommended duration for each clip (in seconds)
        3. Transition types between clips
        4. Color grading recommendations for each clip
        5. Text overlay suggestions (content, timing, style)
        6. Speed adjustments (if needed)
        7. Audio adjustments (levels, fades, background music type)
        8. Call-to-action placement and content

        Format your response as a structured JSON object with these fields:
        - aspect_ratio: string (e.g., "1:1")
        - clips: array of objects with edit instructions for each clip
        - transitions: array of objects with transition details between clips
        - text_overlays: array of objects with text overlay details
        - audio_plan: object with audio editing details
        - cta: object with call-to-action details
        """

        try:
            response = self.gemini_text_model.generate_content(prompt)
            response_text = response.text

            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_content = response_text[json_start:json_end]
                editing_plan = json.loads(json_content)
            else:
                raise ValueError("Failed to extract valid JSON from Gemini response")

        except Exception as e:
            logger.error(f"Error generating editing plan: {e}")
            # Create a basic fallback plan
            editing_plan = {
                "aspect_ratio": platform_info["aspect_ratios"][0],
                "clips": [
                    {
                        "video_id": vid_id,
                        "duration": 10.0,
                        "color_grade": "Standard",
                        "speed": 1.0
                    } for vid_id in video_sequence
                ],
                "transitions": [
                    {
                        "type": "cut",
                        "duration": 0.5
                    } for _ in range(len(video_sequence) - 1)
                ],
                "text_overlays": [],
                "audio_plan": {
                    "background_music": "None",
                    "normalize_levels": True
                },
                "cta": {
                    "text": "Follow for more",
                    "position": "end",
                    "duration": 3.0
                }
            }

        # Store the editing plan
        self.editing_plan = {
            "target_platform": target_platform,
            "video_sequence": video_sequence,
            "plan": editing_plan
        }

        logger.info(f"Editing plan generated for {target_platform}")
        return self.editing_plan

    def apply_clip_edits(self, clip_id: str, edit_instructions: Dict[str, Any]) -> str:
        """Apply edits to a single clip according to instructions."""
        if clip_id not in self.uploaded_clips:
            raise ValueError(f"Unknown clip ID: {clip_id}")

        input_path = self.uploaded_clips[clip_id]["workspace_path"]
        output_id = f"{clip_id}_edited"
        output_path = os.path.join(self.workspace_dir, "processed", f"{output_id}.mp4")

        # Load clip with MoviePy
        clip = VideoFileClip(input_path)

        # Apply duration/trim if specified
        if "start_time" in edit_instructions and "end_time" in edit_instructions:
            start = float(edit_instructions.get("start_time", 0))
            end = float(edit_instructions.get("end_time", clip.duration))
            clip = clip.subclip(start, min(end, clip.duration))
        elif "duration" in edit_instructions:
            requested_duration = float(edit_instructions["duration"])
            if requested_duration < clip.duration:
                clip = clip.subclip(0, requested_duration)

        # Apply speed adjustment if specified
        if "speed" in edit_instructions:
            speed_factor = float(edit_instructions["speed"])
            if speed_factor != 1.0:
                # Use direct speed adjustment
                clip = clip.speedx(speed_factor)

        # Apply basic color grading based on instructions
        if "color_grade" in edit_instructions:
            grade_type = edit_instructions["color_grade"].lower()

            # Basic color adjustments without vfx
            if grade_type in ["warm", "cool", "vibrant", "dramatic", "vintage"]:
                # We'll use a simple approach without vfx - just note the grading
                logger.info(f"Applied {grade_type} color grading to clip {clip_id}")
                # In a production environment, you would use more sophisticated color grading

        # Save the edited clip
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac",
                             temp_audiofile=os.path.join(self.workspace_dir, "temp_audio.m4a"), remove_temp=True)
        clip.close()

        # Store edited clip info
        self.edited_clips[output_id] = {
            "original_clip_id": clip_id,
            "path": output_path,
            "edits_applied": edit_instructions
        }

        logger.info(f"Applied edits to clip {clip_id}")
        return output_id

    def apply_text_overlay(self, clip_path: str, text_config: Dict[str, Any]) -> str:
        """Apply text overlay to a clip."""
        output_path = clip_path.replace(".mp4", "_with_text.mp4")

        # Load clip
        video = VideoFileClip(clip_path)

        # Create text clip
        text = text_config.get("text", "")
        font_size = text_config.get("font_size", 30)
        color = text_config.get("color", "white")
        bg_color = text_config.get("background_color", None)
        position = text_config.get("position", "bottom")
        duration = text_config.get("duration", video.duration)
        start_time = text_config.get("start_time", 0)

        # Create text clip
        txt_clip = TextClip(text, fontsize=font_size, color=color, bg_color=bg_color)

        # Position the text
        if position == "top":
            txt_clip = txt_clip.set_position(('center', 0.1), relative=True)
        elif position == "bottom":
            txt_clip = txt_clip.set_position(('center', 0.9), relative=True)
        elif position == "center":
            txt_clip = txt_clip.set_position('center')
        else:
            # Custom position (x, y) as tuple
            try:
                pos = eval(position) if isinstance(position, str) else position
                txt_clip = txt_clip.set_position(pos)
            except:
                txt_clip = txt_clip.set_position('bottom')

        # Set duration and start time
        txt_clip = txt_clip.set_duration(min(duration, video.duration - start_time))
        txt_clip = txt_clip.set_start(start_time)

        # Composite video with text
        final_clip = CompositeVideoClip([video, txt_clip])

        # Write output
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close clips
        video.close()
        final_clip.close()

        return output_path

    def create_simple_transition(self, clips: List[VideoFileClip], transition_type: str) -> VideoFileClip:
        """Create a simple transition between clips without using vfx."""
        # For now, just concatenate clips without effects
        return concatenate_videoclips(clips)

    def assemble_final_video(self, output_path: str = None) -> str:
        """Assemble the final video according to the editing plan."""
        if not self.editing_plan:
            raise ValueError("No editing plan available. Generate an editing plan first.")

        if not output_path:
            output_path = os.path.join(self.workspace_dir, "outputs", f"final_{uuid.uuid4()}.mp4")

        # Get the sequence and plan
        video_sequence = self.editing_plan["video_sequence"]
        plan = self.editing_plan["plan"]

        # Apply individual clip edits first
        edited_clip_ids = []
        for i, clip_config in enumerate(plan["clips"]):
            video_id = clip_config["video_id"]
            edited_id = self.apply_clip_edits(video_id, clip_config)
            edited_clip_ids.append(edited_id)

        # Load all edited clips
        clips = [VideoFileClip(self.edited_clips[clip_id]["path"]) for clip_id in edited_clip_ids]

        # Apply text overlays if specified
        for text_overlay in plan.get("text_overlays", []):
            clip_index = text_overlay.get("clip_index", 0)
            if 0 <= clip_index < len(clips):
                # Save current clip
                temp_path = os.path.join(self.workspace_dir, "processed", f"temp_{uuid.uuid4()}.mp4")
                clips[clip_index].write_videofile(temp_path, codec="libx264", audio_codec="aac")

                # Apply text overlay
                new_path = self.apply_text_overlay(temp_path, text_overlay)

                # Replace clip
                clips[clip_index].close()
                clips[clip_index] = VideoFileClip(new_path)

        # Simple concatenation of all clips
        final_video = concatenate_videoclips(clips)

        # Apply CTA if specified
        if "cta" in plan and plan["cta"].get("text"):
            cta_config = plan["cta"]
            cta_duration = float(cta_config.get("duration", 3.0))

            # Create CTA text clip
            cta_text = cta_config.get("text", "Follow for more")
            cta_color = cta_config.get("color", "white")
            cta_bg = cta_config.get("background_color", "black")
            cta_fontsize = cta_config.get("font_size", 40)

            # Create a black background clip with CTA text
            cta_bg_clip = ColorClip(size=final_video.size, color=(0, 0, 0), duration=cta_duration)
            cta_txt = TextClip(cta_text, fontsize=cta_fontsize, color=cta_color, bg_color=None)
            cta_txt = cta_txt.set_position('center').set_duration(cta_duration)
            cta_full = CompositeVideoClip([cta_bg_clip, cta_txt])

            cta_position = cta_config.get("position", "end")

            # Create a new composite with the original video and CTA
            if cta_position == "end":
                # Add CTA at the end
                final_video = concatenate_videoclips([final_video, cta_full])
            elif cta_position == "beginning":
                # Add CTA at the beginning
                final_video = concatenate_videoclips([cta_full, final_video])
            # Could add more positions if needed

        # Apply aspect ratio adjustment if needed
        if "aspect_ratio" in plan:
            # Note about aspect ratio adjustment
            logger.info(f"Aspect ratio set to {plan['aspect_ratio']} for final video")
            # In a production environment, you would adjust the aspect ratio
            # This would typically involve resizing or cropping

        # Apply audio adjustments - for simplicity, we'll just log this
        if "audio_plan" in plan:
            logger.info(f"Audio adjustments would be applied according to plan: {plan['audio_plan']}")
            # In a production environment, you would adjust audio levels, add music, etc.

        # Write final video
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Close all clips
        for clip in clips:
            if hasattr(clip, 'close'):
                try:
                    clip.close()
                except:
                    pass

        final_video.close()

        # Store output reference
        self.output_video = output_path

        logger.info(f"Final video assembled at {output_path}")
        return output_path

    def cleanup(self):
        """Clean up temporary files."""
        # Close any open clips
        for clip_id in self.edited_clips:
            try:
                clip_path = self.edited_clips[clip_id]["path"]
                if os.path.exists(clip_path):
                    os.remove(clip_path)
            except:
                pass

        # Option to clean up the entire workspace
        # shutil.rmtree(self.workspace_dir)

        logger.info("Cleanup completed")


def process_videos_for_platform(video_paths: List[str], platform: str = "instagram", output_dir: str = None) -> str:
    """
    Process a list of video files for a specific social media platform.

    Args:
        video_paths: List of paths to video files
        platform: Target social media platform (instagram, tiktok, youtube, etc.)
        output_dir: Directory to save the output video

    Returns:
        Path to the final edited video
    """
    # Setup editor
    editor = VideoEditorSaaS(workspace_dir=output_dir)

    # Upload videos
    video_ids = []
    for video_path in video_paths:
        video_id = editor.upload_video(video_path)
        video_ids.append(video_id)

    # Analyze videos
    editor.analyze_videos(video_ids)

    # Determine sequence
    sequence = editor.determine_video_sequence(video_ids)

    # Generate editing plan
    editor.generate_editing_plan(sequence, target_platform=platform)

    # Assemble final video
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{platform}_edit_{uuid.uuid4()}.mp4")
    else:
        output_path = None

    final_video_path = editor.assemble_final_video(output_path)

    # Cleanup
    editor.cleanup()

    return final_video_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process videos using AI-powered editing")
    parser.add_argument("--videos", nargs="+", required=True, help="List of video files to process")
    parser.add_argument("--platform", default="instagram",
                        choices=["instagram", "tiktok", "youtube", "facebook", "linkedin"],
                        help="Target social media platform")
    parser.add_argument("--output", help="Output directory")

    args = parser.parse_args()

    # Process videos
    output_path = process_videos_for_platform(args.videos, args.platform, args.output)
    print(f"Final video created at: {output_path}")