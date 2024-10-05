import asyncio
import aiohttp
import os
import sys
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, JSONResponse
from dotenv import load_dotenv
from loguru import logger

# Existing imports from your agent worker code
from pipecat.frames.frames import Frame, LLMMessagesFrame, MetricsFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.azure import AzureTTSService, AzureLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer
from pipecat.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from runner import configure

# Load environment variables
load_dotenv(override=True)

# Set up logging
# logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Initialize FastAPI app
app = FastAPI()

# CORS middleware to allow cross-origin requests (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models
class BotRequest(BaseModel):
    bot_name: str
    room_id: str

class BotResponse(BaseModel):
    room_url: str
    token: str

class TextRequest(BaseModel):
    text: str

# WebSocket connections storage: {room_id: [WebSocket, ...]}
active_connections = {}

# Function to broadcast messages to a specific room
async def broadcast_message(room_id: str, message: str):
    """Broadcast a message to a specific room."""
    if room_id in active_connections:
        for connection in active_connections[room_id]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to WebSocket: {e}")

# Define MetricsLogger FrameProcessor
class MetricsLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, MetricsFrame):
            logger.debug(
                f"!!! MetricsFrame: {frame}, ttfb: {frame.ttfb}, processing: {frame.processing}, tokens: {frame.tokens}, characters: {frame.characters}"
            )
        await self.push_frame(frame, direction)

# Active bot set to track running bots: {bot_name: room_id}
active_bots = {}

# Async function to handle bot logic
async def run_bot(room_url: str, token: str, bot_name: str, room_id: str):
    async with aiohttp.ClientSession() as session:
        # Create the DailyTransport with the provided room URL, token, and bot name
        transport = DailyTransport(
            room_url,
            token,
            bot_name,
            DailyParams(
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            ),
        )

        def my_callback(raw_text: str):
            logger.debug(f"Raw Text Received: {raw_text}")
            # Broadcast the received text to the specific room
            asyncio.create_task(broadcast_message(room_id, raw_text))

        tts = AzureTTSService(
            api_key=os.getenv("AZURE_SPEECH_API_KEY"),
            region=os.getenv("AZURE_SPEECH_REGION"),
            voice="en-US-AvaMultilingualNeural",
            callback=my_callback,
        )

        llm = AzureLLMService(
            api_key=os.getenv("AZURE_CHATGPT_API_KEY"),
            endpoint=os.getenv("AZURE_CHATGPT_ENDPOINT"),
            model=os.getenv("AZURE_CHATGPT_MODEL"),
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        # Initialize the pipeline with all necessary components
        pipeline = Pipeline([
            transport.input(),
            LLMUserResponseAggregator(messages),
            llm,
            tts,
            MetricsLogger(),
            transport.output(),
            LLMAssistantResponseAggregator(messages),
        ])

        task = PipelineTask(pipeline, PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ))

        runner = PipelineRunner()

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant['id']}")
            await task.queue_frame(EndFrame())

        # Event handler for the first participant joining the room
        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            messages.append(
                {"role": "system", "content": "Please introduce yourself to the user."}
            )
            await task.queue_frames([LLMMessagesFrame(messages)])

        # Run the pipeline task
        await runner.run(task)

# FastAPI route to start the bot and return room details
@app.post("/start-bot", response_model=BotResponse)
async def start_bot(request: BotRequest):
    try:
        if request.bot_name in active_bots:
            raise HTTPException(status_code=400, detail=f"Bot '{request.bot_name}' is already in the room.")

        active_bots[request.bot_name] = request.room_id
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session, request.room_id)

        # Start the agent (run the bot asynchronously)
        asyncio.create_task(run_bot(room_url, token, request.bot_name, request.room_id))

        # Return the room URL and token
        return BotResponse(room_url=room_url, token=token)
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(status_code=500, detail="Failed to start the bot")

# FastAPI route to configure and start a default bot (optional)
@app.post("/configure", response_class=JSONResponse)
async def configure_endpoint(request: BotRequest):
    try:
        return await start_bot(request)
    except Exception as e:
        logger.error(f"Error in configure_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to configure the bot")

# WebSocket endpoint to handle real-time communication
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str):
    await websocket.accept()
    if room_id not in active_connections:
        active_connections[room_id] = []
    active_connections[room_id].append(websocket)
    logger.info(f"WebSocket connection established for room: {room_id}")
    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received message from WebSocket: {data}")
            # Optionally, you can process incoming messages here
            await broadcast_message(room_id, data)
    except WebSocketDisconnect:
        active_connections[room_id].remove(websocket)
        logger.info(f"WebSocket connection closed for room: {room_id}")
        if not active_connections[room_id]:
            del active_connections[room_id]

# Broadcast message function is already defined above

# Define a response for the home route
@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>WebSocket and Bot Integration</title>
            <script>
                var room_id = "specific_room_id"; // Change as needed
                var ws = new WebSocket(`ws://localhost:8000/ws/${room_id}`);
                
                ws.onmessage = function(event) {
                    var messages = document.getElementById("messages");
                    var message = document.createElement("div");
                    message.textContent = event.data;
                    messages.appendChild(message);
                };
                
                function sendMessage() {
                    var input = document.getElementById("messageInput").value;
                    ws.send(input);
                }
            </script>
        </head>
        <body>
            <h1>WebSocket and Bot Integration Test</h1>
            <input id="messageInput" type="text" placeholder="Type a message...">
            <button onclick="sendMessage()">Send</button>
            <div id="messages"></div>
        </body>
    </html>
    """

# Define the process_text endpoint
@app.post("/process_text/", response_class=JSONResponse)
async def process_text(request: TextRequest):
    try:
        logger.debug(f"Received text for processing: {request.text}")
        # Implement additional logic to send the text through your agent worker if needed.
        return {"message": "Text processed successfully", "input_text": request.text}
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail="Error processing text.")



# Run FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
