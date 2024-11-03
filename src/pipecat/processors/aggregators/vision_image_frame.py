#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from pipecat.frames.frames import Frame, InputImageRawFrame, TextFrame, VisionImageRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
import base64
import time
import cloudinary
import cloudinary.uploader

def upload_base64_to_cloudinary(base64_string, cloud_name, api_key, api_secret):
    """
    Upload a raw base64 string to Cloudinary
    
    Args:
        base64_string (str): The raw base64 string (without data URI prefix)
        cloud_name (str): Cloudinary cloud name
        api_key (str): Cloudinary API key
        api_secret (str): Cloudinary API secret
    
    Returns:
        dict: Contains 'success' (bool), 'url' (str) if successful, 'error' (str) if failed,
              and 'duration' (float) time taken in seconds
    """
    # Configure Cloudinary
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret
    )
    
    start_time = time.time()
    try:
        response = cloudinary.uploader.upload(
            f"data:image/jpeg;base64,{base64_string}",  # Add data URI prefix for Cloudinary
            resource_type="auto"
        )
        
        duration = time.time() - start_time
        return {
            'success': True,
            'url': response.get('secure_url'),
            'duration': round(duration, 2)
        }
        
    except Exception as e:
        duration = time.time() - start_time
        return {
            'success': False,
            'error': str(e),
            'duration': round(duration, 2)
        }

class VisionImageFrameAggregator(FrameProcessor):
    """This aggregator waits for a consecutive TextFrame and an
    InputImageRawFrame. After the InputImageRawFrame arrives it will output a
    VisionImageRawFrame.

    >>> from pipecat.frames.frames import ImageFrame

    >>> async def print_frames(aggregator, frame):
    ...     async for frame in aggregator.process_frame(frame):
    ...         print(frame)

    >>> aggregator = VisionImageFrameAggregator()
    >>> asyncio.run(print_frames(aggregator, TextFrame("What do you see?")))
    >>> asyncio.run(print_frames(aggregator, ImageFrame(image=bytes([]), size=(0, 0))))
    VisionImageFrame, text: What do you see?, image size: 0x0, buffer size: 0 B

    """

    def __init__(self, context):
        super().__init__()
        self._describe_text = None
        self._context  = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            self._describe_text = frame.text
        elif isinstance(frame, InputImageRawFrame):
            if self._describe_text:
                
                frame = VisionImageRawFrame(
                    text=self._describe_text,
                    image=frame.image,
                    size=frame.size,
                    format=frame.format,
                    context=self._context
                )
                await self.push_frame(frame)
                self._describe_text = None
        else:
            await self.push_frame(frame, direction)
