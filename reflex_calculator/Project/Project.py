import reflex as rx


class CalculatorState(rx.State):
    display: str = "0"
    first_number: float = 0
    operator: str = ""
    new_number: bool = True

    def number(self, value: str):
        if self.new_number or self.display == "0":
            self.display = value
            self.new_number = False
        else:
            self.display += value

    def operation(self, op: str):
        self.first_number = float(self.display)
        self.operator = op
        self.new_number = True

    def equals(self):
        second_number = float(self.display)

        if self.operator == "+":
            result = self.first_number + second_number
        elif self.operator == "-":
            result = self.first_number - second_number
        elif self.operator == "*":
            result = self.first_number * second_number
        elif self.operator == "/":
            if second_number == 0:
                self.display = "Error"
                return
            result = self.first_number / second_number
        else:
            return

        self.display = str(result)
        self.new_number = True
        self.operator = ""

    def clear(self):
        self.display = "0"
        self.first_number = 0
        self.operator = ""
        self.new_number = True


def index():
    return rx.center(
        rx.vstack(
            rx.heading("Calculator"),

            rx.text(
                CalculatorState.display,
                font_size="2em",
            ),

            rx.hstack(
                rx.button("7", on_click=CalculatorState.number("7")),
                rx.button("8", on_click=CalculatorState.number("8")),
                rx.button("9", on_click=CalculatorState.number("9")),
                rx.button("/", on_click=CalculatorState.operation("/")),
            ),

            rx.hstack(
                rx.button("4", on_click=CalculatorState.number("4")),
                rx.button("5", on_click=CalculatorState.number("5")),
                rx.button("6", on_click=CalculatorState.number("6")),
                rx.button("*", on_click=CalculatorState.operation("*")),
            ),

            rx.hstack(
                rx.button("1", on_click=CalculatorState.number("1")),
                rx.button("2", on_click=CalculatorState.number("2")),
                rx.button("3", on_click=CalculatorState.number("3")),
                rx.button("-", on_click=CalculatorState.operation("-")),
            ),

            rx.hstack(
                rx.button("0", on_click=CalculatorState.number("0")),
                rx.button("C", on_click=CalculatorState.clear),
                rx.button("=", on_click=CalculatorState.equals),
                rx.button("+", on_click=CalculatorState.operation("+")),
            ),
        )
    )


app = rx.App()
app.add_page(index)