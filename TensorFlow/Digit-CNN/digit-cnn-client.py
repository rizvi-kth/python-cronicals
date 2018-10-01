
# importing the requests library
import requests
import json
import pprint
import base64

def EncodeImageB64():
    # Encode base64 the picture

    input_image = open("seven.png", "rb").read()
    # Encode image in b64
    encoded_input_string = base64.b64encode(input_image)
    input_string = encoded_input_string.decode("utf-8")
    print("Base64 encoded string: " )
    print(input_string)
    print()
    return input_string


img_b64 = EncodeImageB64()
# Sample valid string
# 'iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAYAAAByDd+UAAAABHNCSVQICAgIfAhkiAAAAd5JREFUSIntlU+rOVEcxh8/P4tbNoiyEKWUjSRq8i9lR1bjLcxqlvZ2Nt6AV8BLsLO2wGIWmvxNSKQQFiOc792pW3fcMbir+9TZnPM859PTzDnHAIDwi/r3m7A/4B9Ql/6rLeTzeQiCgMViAUVRUK1WsVwuMRwOnwIaoHIOx+MxPB7Pl7nD4YBut3t3w/l8jnK5jHa7/e26akNBEBAIBCDLMvx+P0KhEFKpFDiOw2w2g8vlunkvlwvW6zWcTicAYDqdqgJVG34ni8WCYDCITqeDSCRym1cUBf1+H7Isw2q1QhRFVCoV1X3oFYPnebperyRJElmt1nve52EOh4NWqxUREfE8f9f7kmMhiiLsdju22y16vd6P/qfaxWIxOp1OxBijZDL5o//phplMBiaTCY1GA81mU1NGd7uPjw/qdDqkKApFo1GtOf3AYrFIjDGq1+uP5PTBstksnc9n2u12xHHce4E2m41GoxExxqhWqz2afwxmNBqp1WoRY4wGgwF5vd73An0+HzHGiDFGuVxOz+fQbna73TSZTIgxRoVCgQwGw3uBpVLp1i4cDuv62TQD4/E47ff7p4Gab5pEIgGz2QwAGI1GOB6PWqNfpPoAq0mSJKTTaWw2G13Ahx7gV+gTRo0HGpCkFMQAAAAASUVORK5CYII='
data ={
    'instances':
        [
            {'b64': img_b64}
        ]
}
jsonData = json.dumps(data)
print("Request: >>> ")
print(jsonData)
print()


# defining the api-endpoint
API_ENDPOINT = "http://localhost:8501/v1/models/my_mnist:predict"
headers = {'content-type': 'application/json'}
r = requests.post(url = API_ENDPOINT, data = jsonData, headers=headers)

print("Response: >>> ")
pprint.pprint(dict(r.headers))
# extracting response text
response_text = r.text
print("JSON body: ")
print(response_text)
output = r.json()
print("The result is: %s " %output['predictions'][0]['classes'])




# from urllib.parse import urlencode
# from urllib.request import Request, urlopen
# Test a normal URL
# url = 'https://httpbin.org/post' # Set destination URL here
# post_fields = {'foo': 'bar'}     # Set POST fields here
#
# request = Request(url, urlencode(post_fields).encode())
# json = urlopen(request).read().decode()
# print(json)
