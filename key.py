import keyboard


## 监控键盘
def int_keyboard(event):
    print(event.name)

# keyboard.on_press(int_keyboard)
# keyboard.wait("q") #使程序进入监听状态，等待按键事件(并结束)

## 监听指定按键
def on_key_pressed(event):
    if event.name == 'esc':  # 监控按下的a键
        print('esc键被按下')
        return True

def on_key_released(event):
    if event.name == 'esc':  # 监控释放的a键
        print('esc键被释放')
        return True

def esc_quit(quit):

    keyboard.on_press_key('esc', on_key_pressed)
    # keyboard.on_release_key('esc', on_key_released)
    keyboard.wait('esc')  # 等待按下esc键后停止监听 时机==按下 》》释放
    quit = False
    return quit

# print(esc_quit())

## 监听组合键
# while True:
#     if keyboard.is_pressed('leftwin') and keyboard.is_pressed('r'):
#         print('win+R被按下')
#         break
