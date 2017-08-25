class Schedule:
    def __init__(self):
        self.bookings = []

    def book(self, name):
        if len(self.bookings) < 3: 
            self.bookings.append(name)
            print("Booked ")
            print(self.bookings)
        else:
            print("Booking not possible ")
        

x = Schedule()
print(x)
x.book("Rizvi")
x.book("Raj")
x.book("Prem")
x.book("Sam")

