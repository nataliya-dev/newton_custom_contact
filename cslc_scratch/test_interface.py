import newton
for name in dir(newton.viewer.ViewerGL):
    low = name.lower()
    if any(k in low for k in ['particle', 'show', 'render', 'option', 'flag', 'toggle', 'display']):
        print(f'  ViewerGL.{name}')