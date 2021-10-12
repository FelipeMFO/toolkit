
def read_file(file_path):
    """Read file content on file_path."""
    fd = open(file_path, 'r')
    content = fd.read()
    fd.close()

    return content
