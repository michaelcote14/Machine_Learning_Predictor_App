from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.stacklayout import StackLayout

class ModelRunner(GridLayout):
    pass

class BoxLayoutExample(BoxLayout):
    def on_toggle_button_state(self, toggle_widget):
        print('Toggle State: ' + toggle_widget.state) # gets a down or normal state with each click
        if toggle_widget.state == 'normal':
            toggle_widget.text = 'Start'
        else:
            toggle_widget.text = 'Stop'


class MachineLearningApp(App):
    pass

class MainMenu(BoxLayout):
    pass

class Options(StackLayout):
    pass


class MainWidget(Widget):
    pass

MachineLearningApp().run() # this runs the app