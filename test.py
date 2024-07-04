from voyager import Voyager

openai_api_key = "YOUR_API_KEY"

voyager = Voyager(
    mc_port=25565,
    openai_api_key=openai_api_key,
)

# start lifelong learning
voyager.learn()