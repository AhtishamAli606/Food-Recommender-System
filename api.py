import webbrowser

from flask import Flask, jsonify, request, render_template
import recommendersystem

app = Flask(__name__, template_folder='Template')


@app.route('/api/dishes', methods=['GET'])
def get_dishes():
    # Call the function to get the dish info
    response_data = recommendersystem.get_dish_info()
    print(response_data)

    # Render the results.html template with the response data
    rendered_template = render_template('results.html', result=response_data)

    return rendered_template


@app.route('/api/dishes', methods=['GET', 'POST'])
def user_input_recommendations():
    if request.method == 'POST':
        # Get the dish input from the form submission
        dish_name = request.form.get('dish_name')

        # Call the function to get the dish info
        result = recommendersystem.get_dish_input(dish_name)
        print(result)

        if 'error' in result:
            # Dish not found in the dataset
            return jsonify({'error': 'Dish not found in the dataset.'}), 404
        else:
            # Render the HTML template with the result
            return render_template('results.html', result=result)

    else:
        # Get all dish names
        dish_names_low_bias, _ = recommendersystem.get_dish_info()

        # Render the HTML form for user input with all dish names
        return render_template('input_form.html', dish_names=dish_names_low_bias)


if __name__ == '__main__':
    app.run(debug=True)
