from TikTokApi import TikTokApi

def downloadVideos():
    videoID = []
    with TikTokApi() as api:
        tagname = open("hashtag.txt").read() 
        count = open("count.txt").read() 
        stop = int(count)   
        count = int(count) + 20
        x = 0
        videocount = count
        tag = api.hashtag(name=tagname)
        for video in tag.videos(count=videocount):
            videoID.append(video.id)
            print(video.id)

        for id in videoID:
            try:
                if x == stop:
                    break
                video = api.video(id=id)
                video_data = video.bytes()
                with open(id+".mp4", "wb") as out_file:
                    out_file.write(video_data)
                    x += 1
                
            except:
                print("Something went wrong")
