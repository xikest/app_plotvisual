import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
import requests
from io import BytesIO




def download_image_example(url="https://raw.githubusercontent.com/xikest/app_plotvisual/main/sample_img.png", demo_mode=False):

    try:
        response = requests.get(url)
    # with open("downloaded_image.png", "wb") as f:
    #     f.write(response.content)

    # 이미지 데이터를 바이트로 변환
        img_bytes = BytesIO(response.content).read()
    except:
        st.error("http connection error")
        img_bytes = ""


    # img_file = Image.open(BytesIO(response.content))
    # st.image(img_file, caption="Downloaded Image", use_column_width=True)
    if demo_mode:
        return img_bytes
    else:
    # 다운로드 버튼에 이미지 데이터 설정
        st.download_button(
                label="Sample download",
                data=img_bytes,
                file_name="example.png",
                mime="image/png",
                key="download_sample_img"
            )
        return None
def to_lab_image(image_data, input_type='BGR', byte_type=False):
    if byte_type:
        image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
    else:
        image_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)


    conversion = cv2.COLOR_BGR2Luv if input_type == 'BGR' else cv2.COLOR_RGB2Luv
    image_LUV = cv2.cvtColor(image, conversion)

    y, x, z = image_LUV.shape
    LUV_flat = np.reshape(image_LUV, [y * x, z])

    colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if input_type == 'BGR' else image
    colors = np.reshape(colors, [y * x, z]) / 255.

    l = LUV_flat[:, 0]
    u = LUV_flat[:, 1]
    v = LUV_flat[:, 2]



    # # Create a Streamlit figure for the input image
    # fig_input = plt.figure(figsize=(6, 6))
    # ax_input = fig_input.add_subplot(111)
    # ax_input.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Display the input image
    # ax_input.set_title('Input Image')
    # ax_input.axis('off')

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(xs=u, ys=v, zs=l, s=10, c=colors, lw=0)
    ax1.set_xlabel('U')
    ax1.set_ylabel('V')
    ax1.set_zlabel('L')
    ax1.set_xlim([0, 255])
    ax1.set_ylim([0, 255])
    ax1.set_zlim([0, 255])
    ax1.set_xticks([])  # X 축 눈금 비활성화
    ax1.set_yticks([])  # Y 축 눈금 비활성화
    ax1.set_zticks([])  # Z 축 눈금 비활성화
    ax1.set_title('L-U-V')

    ax2 = fig.add_subplot(222)
    ax2.scatter(y=v, x=u, s=10, c=colors, lw=0)
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.set_xlim([0, 255])
    ax2.set_ylim([0, 255])
    ax2.set_xticks([])  # X 축 눈금 비활성화
    ax2.set_yticks([])  # Y 축 눈금 비활성화
    ax2.set_title('u-v')


    ax3 = fig.add_subplot(223)
    ax3.scatter(x=u, y=l, s=10, c=colors, lw=0)
    ax3.set_xlabel('U')
    ax3.set_ylabel('L')
    ax3.set_xlim([0, 255])
    ax3.set_ylim([0, 255])
    ax3.set_xticks([])  # X 축 눈금 비활성화
    ax3.set_yticks([])  # Y 축 눈금 비활성화
    ax3.set_title('L-A')


    ax4 = fig.add_subplot(224)
    ax4.scatter(x=v, y=l, s=10, c=colors, lw=0)
    ax4.set_xlabel('V')
    ax4.set_ylabel('L')
    ax4.set_xlim([0, 255])
    ax4.set_ylim([0, 255])
    ax4.set_xticks([])  # X 축 눈금 비활성화
    ax4.set_yticks([])  # Y 축 눈금 비활성화
    ax4.set_title('L-V')


    # Streamlit에 그래프 추가
    # st.pyplot(fig_input)
    st.pyplot(fig)