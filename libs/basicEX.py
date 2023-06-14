class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append('Key not found: {}'.format(keys))
        if visited is not None:
            messages.append('Visited: {}'.format(visited))
        messages.append('Cause:\n{}'.format(cause))
        message = '\n'.join(messages)
        super().__init__(message)