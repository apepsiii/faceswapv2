import midtransclient

core_api = midtransclient.CoreApi(
    is_production=True,  # Ubah ke True saat deploy live
    server_key='Mid-server-xxxxx',
    client_key='Mid-client-xxxxx'
)