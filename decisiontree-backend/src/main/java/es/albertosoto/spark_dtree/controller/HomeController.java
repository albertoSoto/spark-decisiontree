package es.albertosoto.spark_dtree.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

/**
 * Controller for handling the home page and root URL
 */
@Controller
public class HomeController {

    /**
     * Redirect root URL to index.html
     * @return Redirect to index.html
     */
    @GetMapping("/")
    public String home() {
        return "redirect:/index.html";
    }
}
