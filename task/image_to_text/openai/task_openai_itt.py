import asyncio
import base64
from io import BytesIO
from pathlib import Path

from task._models.custom_content import Attachment, CustomContent
from task._models.message import Message
from task._utils.bucket_client import DialBucketClient
from task._utils.constants import API_KEY, DIAL_CHAT_COMPLETIONS_ENDPOINT, DIAL_URL
from task._utils.model_client import DialModelClient
from task._models.role import Role
from task.image_to_text.openai.message import ContentedMessage, TxtContent, ImgContent, ImgUrl

def call_with_base64(client: DialModelClient, base64_image: str):
    text_content = TxtContent(text="what is on the image, be short")
    image_url = ImgUrl(url=f"data:image/png;base64,{base64_image}")
    img_content = ImgContent(image_url=image_url)
    contented_message = ContentedMessage(content=[text_content, img_content], role=Role.USER)
    response = client.get_completion(messages=[contented_message])
    print(response.content)


async def put_image(image_bytes: bytes) -> Attachment:
    bucket = DialBucketClient(api_key=API_KEY, base_url=DIAL_URL)
    async with bucket as bucket_client:
        file_name = 'dialx-banner.png'
        mime_type_png = 'image/png'
        image_content = BytesIO(image_bytes)

        attachment = await bucket_client.put_file(
            name=file_name,
            mime_type=mime_type_png,
            content=image_content
        )

        return Attachment(
            title=file_name,
            url=attachment.get("url"),
            type=mime_type_png
        )


def call_with_dial_bucket(client: DialModelClient, image_bytes: bytes):
    attachment = asyncio.run(put_image(image_bytes))
    #print(attachment)
    message = Message(
        role=Role.USER,
        content="what is on the image? be short",
        custom_content=CustomContent(attachments=[attachment]))
    completion = client.get_completion([message])
    print(completion.content)
    pass


def start() -> None:
    project_root = Path(__file__).parent.parent.parent.parent
    image_path = project_root / "dialx-banner.png"

    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')

    # TODO:
    #  1. Create DialModelClient
    #  2. Call client to analise image:
    #    - try with base64 encoded format
    #    - try with URL: https://a-z-animals.com/media/2019/11/Elephant-male-1024x535.jpg
    #  ----------------------------------------------------------------------------------------------------------------
    #  Note: This approach embeds the image directly in the message as base64 data URL! Here we follow the OpenAI
    #        Specification but since requests are going to the DIAL Core, we can use different models and DIAL Core
    #        will adapt them to format Gemini or Anthropic is using. In case if we go directly to
    #        the https://api.anthropic.com/v1/complete we need to follow Anthropic request Specification (the same for gemini)

    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name='gpt-4o',
        api_key=API_KEY,
    )
    #call_with_base64(client, base64_image)
    call_with_dial_bucket(client, image_bytes)


start()