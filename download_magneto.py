import numpy as np
from sunpy.net import jsoc
from sunpy.net import attrs as a

series_name = "hmi.ME_720s_fd10"
user_email = "g.samarth@tifr.res.in"
download_dir = "/scratch/seismogroup/data/HMI/magnetograms/"

client = jsoc.JSOCClient()
response = client.search(a.Time('2010/4/30T00:00:00', '2010/5/20T00:00:00'),
                         a.jsoc.Series(series_name),
                         a.jsoc.PrimeKey('LMIN', ell) &
                         a.jsoc.PrimeKey('LMAX', ell) &
                         a.jsoc.PrimeKey('NDT', '138240'),
                         a.jsoc.Notify(user_email))
requests = client.request_data(response)
count = 0
while requests.status > 0:
    time.sleep(3)
    requests = client.request_data(response)
    print(f"request ID = {requests.id}; status = {requests.status}")
    if count > 20:
        print("Wait count = {count}. Trying to download")
        break
    count += 1
print(f" status = {requests.status}: Ready for download")
res = client.get_request(requests, path=download_dir)
