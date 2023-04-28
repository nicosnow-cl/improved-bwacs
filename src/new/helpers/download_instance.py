import vrplib


def download_instance(nane, path):
    try:
        vrplib.download_instance(nane, path)
        vrplib.download_solution(nane, path)

        return True
    except Exception as e:
        print(e)

        return False
