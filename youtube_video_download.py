from pytube import YouTube

def download_video(url, output_path='.'):
    try:
        # 유튜브 동영상 객체 생성
        yt = YouTube(url)

        # 최고 품질의 스트림 얻기
        video_stream = yt.streams.get_highest_resolution()

        # 동영상 다운로드 시작
        print("다운로드 중...")
        video_stream.download(output_path)
        print("다운로드 완료!")

    except Exception as e:
        print(f"다운로드 중 오류 발생: {str(e)}")

# 유튜브 동영상 URL을 지정하여 다운로드 시작
video_url = "https://www.youtube.com/watch?v=Dbj1GNT3Mjk"

download_video(video_url, output_path='down')