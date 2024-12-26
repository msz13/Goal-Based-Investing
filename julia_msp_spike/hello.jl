using Pkg
Pkg.add("Gtk")

using Gtk

win = GtkWindow("My First Gtk.jl Program", 400, 200)

b = GtkButton("Click Me")
push!(win,b)

function on_button_clicked(w)
  println("The button has been clicked")
end
signal_connect(on_button_clicked, b, "clicked")

showall(win)