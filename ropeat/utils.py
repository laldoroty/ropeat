def get_obj_type_from_ID(ID):
    if ID > 9921000000000 and ID < 10779202101973:
        return 'galaxy'
    elif ID > 30328699913 and ID < 50963307358:
        return 'star'
    elif ID > 20000001 and ID < 120026449:
        return 'transient'
    else:
        return 'unknown'

def train_config(objtype):
    if objtype == 'galaxy':
        return 0
    elif objtype == 'star':
        return 1
    elif objtype == 'transient':
        return 2
    else: 
        return 3