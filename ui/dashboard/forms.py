from flask_wtf import FlaskForm
from wtforms.fields import SelectField


# creates category dropdown form
class DropdownForm(FlaskForm):
    category = SelectField("Category", choices=["All Categories", "Category 1", "Category 2", "Category 3", "Category 4"], default="All Categories")




