import asyncio
import concurrent
import numpy as np
import requests
import io

INFERENCE_ENDPOINT = "http://127.0.0.1:8080/predictions/ensenti"


def predict_batch():
    images = ["/img/cat_01.jpg", "/img/cat_02.jpg"]
    buffer = []
    for img in images:
        output = io.BytesIO()
        np.save(output, img)
        output.seek(0, 0)
        buffer.append(output)
    return buffer


def predict_batch_txt():
    texts_ = ["Aäsöop : I am so glad to have you in my home. You are my best friend. I like spending time with my best friend. I am hopping for a good relationship between us.",
              "I dont like their customer support. I was on hold for 40 minutes, their customer support service is a nightmare."]
    buffer = []
    for txt in texts_:
        output = txt.encode("utf-8", "ignore")
        buffer.append(output)
    return buffer

    # loop = asyncio.get_event_loop()
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    # responses = await asyncio.gather(*[loop.run_in_executor(executor, requests.post, INFERENCE_ENDPOINT, pickled_image) for pickled_image in buffer])
    # probs = [res.json() for res in responses]
    # return probs


async def main(loop):
    buffer = predict_batch_txt()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    tasks = [loop.run_in_executor(executor, requests.post, INFERENCE_ENDPOINT, pickled_image) for pickled_image in buffer]
    responses = await asyncio.gather(*tasks)
    probs = [res.text for res in responses if res.status_code == 200]
    # probs = [print(res) for res in responses]
    print(probs)
    return probs


    # urls = ["http://www.irs.gov/pub/irs-pdf/f1040.pdf"]
    # async with aiohttp.ClientSession(loop=loop) as session:
    #     tasks = [download_coroutine(session, url) for url in urls]
    #     await asyncio.gather(*tasks)




if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(loop))