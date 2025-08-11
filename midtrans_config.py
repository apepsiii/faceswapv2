import midtransclient

core_api = midtransclient.CoreApi(
    is_production=True,  # Ubah ke True saat deploy live
    server_key='XXXX',
    client_key='xxxx'
)