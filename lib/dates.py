#################################################################
def format_duration(seconds):
    remainder = seconds
    (hours, remainder) = divmod(remainder, 60*60)
    (minutes, remainder) = divmod(remainder, 60)
    
    if hours > 0:
        return '{:>2}h:{:>2}m:{:>2}s'.format(hours,minutes,remainder)
    elif minutes > 0:
        return '    {:>2}m:{:>2}s'.format(minutes,remainder)
    else:
        return '        {:>2}s'.format(remainder)
