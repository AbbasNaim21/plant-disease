$(document).ready(function () {
  // Handle navigation clicks
  $("#homeLink").click(function () {
    $("#homeSection").show();
    $("#uploadSection").show();
    $("#aboutSection").hide();
  });

  $("#aboutLink").click(function () {
    $("#homeSection").hide();
    $("#uploadSection").hide();
    $("#aboutSection").show();
  });

  $("#uploadForm").on("submit", function (e) {
    e.preventDefault();

    // Clear previous results
    $("#predictionResult").empty();
    $("#loader").removeClass("hidden");

    var formData = new FormData();
    formData.append("file", $("#fileInput")[0].files[0]);

    $.ajax({
      url: "/predict", // Replace with your backend endpoint
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: function (response) {
        $("#loader").addClass("hidden");
        $("#predictionResult").html(
          `<h3>Prediction: ${response.prediction}</h3>`
        );
      },
      error: function () {
        $("#loader").addClass("hidden");
        $("#predictionResult").html(
          `<h3>Error: Unable to analyze the image.</h3>`
        );
      },
    });
  });
});
